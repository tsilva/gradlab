# YouTube publishing credentials

The default YouTube credential backend uses private `youtube_client_secret.json` and `youtube_token.json` files in the GradLab config directory. To select Infisical, install and log in to its CLI, store the complete client and token JSON objects in a dedicated project, and create `youtube_infisical.json` in that same config directory:

```json
{
  "instance": "https://app.infisical.com",
  "project_id": "your-project-UUID",
  "environment": "prod",
  "path": "/youtube",
  "executable": "infisical",
  "secrets": {
    "client": "GRADLAB_YOUTUBE_CLIENT_JSON",
    "token": "GRADLAB_YOUTUBE_TOKEN_JSON"
  }
}
```

Both operator references and YouTube store configurations require `instance`,
`project_id`, `environment` and `path`; none has an implicit default. Existing
Infisical configurations missing these fields must be updated before use. Select
the narrow folder that owns the credentials; `/` is accepted only when explicitly
configured. Malformed scopes and unknown fields stop access before calling the CLI.

Optional `login_profile` pins the Infisical login. Its identity needs read access to both secrets and write access to the token secret. The player, durable publisher, and standalone YouTube script use the selected store, including refresh and consent writeback. Writes use stdin without creating plaintext JSON files. Vault failures stop credential access without a local-file fallback; local token locking remains in place for the player and publisher.

The standalone script also accepts a host-provisioned loopback relay through
`GRADLAB_YOUTUBE_RELAY`, `GRADLAB_YOUTUBE_CHANNEL_ID`, and the public marker
`GRADLAB_YOUTUBE_RELAY_TOKEN=gradlab_youtube_proxy_placeholder`. It checks the
channel before publishing and sends uploads in chunks to local handles supplied
by the relay. The operator supplies the relay explicitly. OAuth setup and
credential storage stay outside this mode.

For a new upload, `--metadata-file` accepts a JSON object with `title`,
`description`, `tags` (a list), and `category_id` (a string). Use it in place
of the inline metadata options when you want to review the draft as a file.

From the repository, reconnect without uploading or changing a video:

```bash
uv run --frozen python scripts/upload_youtube_video.py --auth-only
```

After verifying access and token refresh, retire the former local OAuth files. A Google `deleted_client` error requires a replacement Desktop OAuth client before reauthorization; copying the deleted client's JSON to Infisical cannot restore it.
