# Timeline UI report images

The `*-original.png` files are the original screenshots supplied with the UI report; colored annotations are part of those screenshots. The `*-proposal-*-en.png` files are earlier AI-generated English design concepts, **not screenshots of the implemented build**.

- `001`: clipped gallery style categories and the missing ordinary vertical-wheel path.
- `002`: separation and spacing of the inter-segment controls.
- `003`: missing `global_prompt` field heading. The label-only concept predates its integration into the implemented collapsible heading.
- `004`: double-click preview for segment/Cast thumbnails, including source-video playback. The implementation uses previous/next controls rather than the concept's example thumbnail rail.

The concept captions saying “not yet implemented” reflect their creation time. The implemented UI was subsequently packaged and reported working by the user in their local ComfyUI installation. Automated browser/regression checks are described separately in the PR and `tests/browser/README.md`.

No category-collapse screenshot or concept is included; that improvement is documented in text only at the user's request.
