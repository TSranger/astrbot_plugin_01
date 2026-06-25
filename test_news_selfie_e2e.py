"""End-to-end test script for the news selfie pipeline.

Usage:
    cd AstrBot_project_dir
    uv run python -m astrbot_plugin_01.test_news_selfie_e2e
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import yaml

from astrbot_plugin_01.llm_router import PluginLLMRouter
from astrbot_plugin_01.news_selfie import (
    NewsEvaluator,
    NewsFetcher,
    NewsSelfiePipeline,
    SelfieImageGenerator,
    resolve_bot_appearance,
    resolve_news_data_dir,
)

_PLUGIN_DIR = Path(__file__).resolve().parent
with open(_PLUGIN_DIR / "config.yaml", encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)


def _create_mock_context():
    mock_provider = MagicMock()
    mock_provider.text_chat = MagicMock(
        return_value=MagicMock(completion_text="mock fallback")
    )
    mock_context = MagicMock()
    mock_context.get_using_provider = MagicMock(return_value=mock_provider)
    return mock_context


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


class _DebugRouter(PluginLLMRouter):
    """Router wrapper that logs raw LLM responses."""

    async def text_chat(self, role, prompt, system_prompt, **kwargs):
        result = await super().text_chat(role, prompt, system_prompt, **kwargs)
        print(f"  [DEBUG] text_chat role={role} result_len={len(result)}")
        print(f"  [DEBUG] result[:300]: {result[:300]}")
        return result


async def main():
    _print_separator("News Selfie Pipeline - End-to-End Test")

    news_settings = _CONFIG.get("news_selfie_settings", {})
    llm_settings = _CONFIG.get("llm_settings", {})

    print(f"[config] news_selfie enabled: {news_settings.get('enabled')}")
    print(f"[config] news_source: {news_settings.get('news_source')}")
    print(f"[config] image_gen_provider: {news_settings.get('image_gen_provider')}")
    print(f"[config] llm mode: {llm_settings.get('mode')}")
    print(f"[config] max_headlines: {news_settings.get('max_headlines')}")
    print(f"[config] max_selfies_per_run: {news_settings.get('max_selfies_per_run')}")
    print(f"[config] http_proxy: {news_settings.get('http_proxy', 'none')}")
    print()

    if not news_settings.get("enabled"):
        print("[SKIP] news_selfie_settings.enabled is false")
        return

    mock_context = _create_mock_context()
    router = _DebugRouter(context=mock_context, settings=llm_settings)
    data_dir = resolve_news_data_dir()

    _print_separator("STEP 1: NewsFetcher")
    fetcher = NewsFetcher(news_settings)
    async with aiohttp.ClientSession() as session:
        headlines = await fetcher.fetch_headlines(session)
    print(f"  Fetched {len(headlines)} headlines")
    for i, h in enumerate(headlines):
        print(f"  [{i + 1}] {h['title'][:100]}")
        print(f"       source: {h.get('source', '')}")

    if not headlines:
        print("[FAIL] No headlines fetched")
        return
    print("[PASS] NewsFetcher works")

    _print_separator("STEP 2: NewsEvaluator (LLM)")
    evaluator = NewsEvaluator(router)
    big_news = await evaluator.evaluate_news(headlines)
    print(f"  Big news found: {len(big_news)}")
    for bn in big_news:
        print(f"  - [{bn.get('category')}] {bn['title'][:80]}")

    if not big_news:
        print("[INFO] No big news detected (LLM returned no matchable JSON)")
        print("  Testing LLM router directly...")
        test_result = await router.text_chat(
            role="news_evaluator",
            prompt=(
                "请评估以下新闻：\n"
                "1. <news_title>测试</news_title>\n"
                "<news_summary>一场大地震</news_summary>\n"
                "只返回 JSON 数组。"
            ),
            system_prompt=(
                "你是一个新闻评估专家。返回严格的 JSON 数组：\n"
                '[{"title": "原标题", "is_big_news": true, "reason": "理由", "category": "other"}]\n'
                "只返回 JSON。"
            ),
        )
        print(f"  Direct LLM result: {test_result[:500]}")
    else:
        print("[PASS] NewsEvaluator works")

    _print_separator("STEP 3: BotAppearance Resolution")
    skill_file = _CONFIG.get("skill_settings", {}).get("active_skill_file", "")
    if skill_file:
        resolved = (_PLUGIN_DIR / skill_file).resolve()
        skill_dir = resolved.parent if resolved.is_file() else resolved
    else:
        skill_candidates = sorted(_PLUGIN_DIR.glob("skills/*.skill"))
        skill_dir = skill_candidates[0] if skill_candidates else _PLUGIN_DIR
    print(f"  skill_dir: {skill_dir}")

    appearance = resolve_bot_appearance(skill_dir)
    print(f"  gif: {appearance.reference_gif}")
    print(f"  images: {len(appearance.reference_images)} files")
    for img in appearance.reference_images:
        print(f"    - {img.name}")
    print(f"  text_description: {appearance.text_description[:50]}")

    has_any_appearance = (
        appearance.reference_gif is not None
        or len(appearance.reference_images) > 0
        or bool(appearance.text_description)
    )
    if has_any_appearance:
        print("[PASS] BotAppearance resolved")
    else:
        print("[FAIL] No bot appearance resolved")

    _print_separator("STEP 4: SelfieImageGenerator")
    if not news_settings.get("image_gen_api_key"):
        print("[SKIP] No image_gen_api_key configured")
    else:
        img_gen = SelfieImageGenerator(news_settings, data_dir)
        scene = "新闻现场围观"
        skill_md = skill_dir / "SKILL.md"
        skill_content = skill_md.read_text("utf-8") if skill_md.exists() else ""
        async with aiohttp.ClientSession() as session:
            result_path = await img_gen.generate_selfie(
                appearance,
                scene,
                session,
                skill_content=skill_content,
            )
        if result_path and result_path.exists():
            print(f"  [PASS] Image generated: {result_path}")
            print(f"  File size: {result_path.stat().st_size} bytes")
        else:
            print("[FAIL] Image generation returned no result")

    _print_separator("STEP 5: Full Pipeline")
    print("  Running NewsSelfiePipeline.run()...")
    pipeline = NewsSelfiePipeline(
        settings={**news_settings, "active_skill_file": skill_file},
        router=router,
        plugin_dir=_PLUGIN_DIR,
        data_dir=data_dir,
    )

    start_time = time.time()
    try:
        results = await pipeline.run()
    except Exception as exc:
        print(f"[FAIL] Pipeline raised: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return

    elapsed = time.time() - start_time
    print(f"\n  Elapsed: {elapsed:.1f}s | Results: {len(results)}")
    for i, r in enumerate(results):
        print(f"\n  --- Result {i + 1} ---")
        print(f"  Title: {r.get('news_title', 'N/A')[:80]}")
        print(f"  Category: {r.get('category', 'N/A')}")
        print(f"  Text: {r.get('text', 'N/A')[:150]}")
        print(f"  Image: {r.get('image_path') or 'NOT GENERATED'}")
        print(f"  NewsImage: {r.get('news_image') or 'NONE'}")
    print(f"  Cache entries: {len(pipeline.cache)}")

    if results:
        print("[PASS] Full pipeline completed with results")
    else:
        print("[INFO] No results - see above for reason")

    _print_separator("Test Summary")
    print("  Check the logs above for PASS/FAIL on each step.")
    print("  - NewsFetcher: requires network + proxy (if in China)")
    print("  - NewsEvaluator: requires DeepSeek API key")
    print("  - BotAppearance: local file resolution")
    print("  - SelfieImageGenerator: requires MIMO API key")
    print("  - Full Pipeline: combines all above")


if __name__ == "__main__":
    asyncio.run(main())
