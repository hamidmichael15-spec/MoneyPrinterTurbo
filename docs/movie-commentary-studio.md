# Movie Commentary Studio

This Streamlit page adds an upload-first commentary workflow without changing the existing MoneyPrinterTurbo generation flow.

## Current MVP capabilities

- Upload a real MP4/MOV/MKV/WEBM/M4V clip (maximum 250 MB).
- Enter scene notes and generate a short spoken commentary using the repository's configured LLM provider.
- Edit the generated narration before rendering.
- Generate speech with Edge TTS; no separate TTS API key is required.
- Set original clip audio volume (muted by default).
- Optionally upload background music (maximum 30 MB) and set its volume.
- Convert the clip to vertical 9:16, burn in sentence captions, and export MP4.

## Important current limitation

This MVP does **not** automatically understand the uploaded video's visual content. The user supplies scene notes, and the LLM is instructed not to invent details beyond those notes. Automated frame sampling and vision-model scene analysis are a follow-up feature and require a compatible vision model/provider and its credentials.

## Launch

Run MoneyPrinterTurbo as usual. Streamlit discovers this page from \`webui/pages/\` when the main WebUI is launched:

\`\`\`bash
sh webui.sh
\`\`\`

Open the **Movie Commentary Studio** page from the Streamlit page navigation.

## Runtime requirements

The repository already pins MoviePy 2.x, Streamlit, and Edge TTS in its dependency manifests. FFmpeg must be available through the project's existing MoviePy/imageio-ffmpeg setup. The page uses the currently configured MoneyPrinterTurbo LLM provider for script generation.
