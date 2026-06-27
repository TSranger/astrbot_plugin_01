# AstrBot Plugin 01 Repair Plan

## Goal
Fix three concrete issues in `astrbot_plugin_01`:
1. News selfie sending must not pass `group_id` as a fake session string.
2. Immediate replies must not be truncated into obvious half-sentences.
3. The sanitize layer must preserve valid fallback replies like `...` instead of deleting them.

## Context
Observed symptoms from logs:
- `news_selfie` falls back to `session = group_id`, then `StarTools.send_message()` throws `不合法的 session 字符串`.
- Some immediate replies end as fragments such as `哎呀，当`, which looks like generation-time early stop rather than transport truncation.
- When the model returns `...`, `_sanitize_reply_text()` currently strips it down to empty, causing `Empty reply after sanitize` and suppressing a valid no-op style reply.

Relevant code paths:
- `main.py` tracks real sessions via `self.group_sessions[group_id] = str(event.session)`.
- `main.py::_send_reply()` splits long replies, but does not explain half-sentence outputs.
- `main.py::_build_chat_reply()` constrains replies to about 2 sentences and can return `...` on empty retry.
- `main.py::_sanitize_reply_text()` removes wrappers and punctuation, and currently treats punctuation-only strings as empty.
- `main.py` news selfie cron sends with cached session first, then incorrectly falls back to `group_id`.

## Decisions
- Do not change AstrBot core or `StarTools`.
- Do not rewrite the news pipeline architecture.
- Keep current plugin behavior for normal replies, memory, and skill identity.
- Treat `...` as an intentional fallback reply, not as invalid output.

## Planned Changes
1. Fix news selfie sending session handling.
- Remove the `group_id` fallback in the news selfie send loop.
- Only call `StarTools.send_message()` when a real cached session exists in `self.group_sessions`.
- If no session is available, log a warning and skip that group for this run.
- Keep the current output-silence guard and multi-result send loop intact.

2. Preserve valid fallback replies in sanitization.
- Update `_sanitize_reply_text()` so that a reply consisting only of `...`, `…`, or similar intentional ellipsis forms is returned as-is.
- Preserve short punctuation-only no-op replies when they are a deliberate fallback from generation, while still rejecting truly empty output.
- Keep existing stripping of code fences, wrapper quotes, and obvious JSON shells.
- Avoid broad acceptance of arbitrary punctuation; only allow the minimal safe fallback forms needed for the bot’s no-words behavior.

3. Reduce half-sentence immediate replies.
- Review the chat generation path in `_build_chat_reply()` and the relevant `llm_router.py` defaults for chat output length.
- Add stronger prompt guidance that the reply must be a complete natural sentence or an intentional ellipsis fallback, not a broken fragment.
- If the generated text is clearly incomplete or ends in a dangling starter phrase, retry once before sending.
- Keep the retry logic lightweight and local to the existing reply flow.

## Implementation Notes
- The session fix should follow the existing pattern used when registering `self.group_sessions[group_id] = str(event.session)`.
- The sanitize adjustment should be narrowly scoped so it does not start accepting malformed JSON or noisy bracketed output.
- If a retry returns `...`, the sender should be allowed to send it instead of treating it as empty.
- The half-sentence mitigation should not over-constrain legitimate short replies such as single-word answers.

## Validation
- Trigger a news selfie task in a group that has a cached session and confirm it sends successfully.
- Trigger a news selfie task in a group without a cached session and confirm it logs a warning instead of raising a session parsing error.
- Trigger an immediate reply where the model returns `...` and confirm the plugin sends it instead of dropping it.
- Trigger a normal immediate reply and confirm it is no longer cut off into a visible fragment.
- Check logs for the absence of `不合法的 session 字符串` and `Empty reply after sanitize` for deliberate ellipsis replies.

## Risks
- Removing the `group_id` fallback means some startup runs may skip news selfie delivery until sessions are cached, but this is preferable to invalid sends.
- Loosening sanitize rules too much could let through malformed output; the allowance must stay limited to intentional ellipsis forms.
- If the upstream model itself is unstable, prompt and retry changes may reduce but not completely eliminate fragment replies.

## Out of Scope
- Changing the news article extraction strategy.
- Modifying skill content or persona rules.
- Changing AstrBot core session semantics.
- Reworking the plugin’s memory or followup architecture.
