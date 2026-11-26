# Publish this repository on GitHub

The generated folder is ready to become a repository. It has **not** been published automatically.

1. Review the English and Chinese README, license and research status.
2. Add your name/contact or a profile link if desired. Do not add degrees, affiliations, results or acceptance claims that are not yours.
3. Create an empty GitHub repository, for example `semantic-shift-lab`, using your preferred visibility. Do not initialize a second README remotely.
4. In this folder, run:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Add reproducible semantic invariance and sensitivity research pilot"
git remote add origin https://github.com/YOUR_USERNAME/semantic-shift-lab.git
git push -u origin main
```

Replace `YOUR_USERNAME`. Authentication uses your GitHub tooling; never put a token in source files or the remote URL. A `.gitignore` excludes local runs, checkpoints and environments. Explicit fixture files under `examples/` are safe to include because they are clearly synthetic demonstrations.

Suggested repository description:

> A reproducible LLM research pilot for answer invariance, fact sensitivity, controlled SFT, and failure analysis.

Suggested topics: `llm`, `nlp`, `evaluation`, `robustness`, `lora`, `reproducibility`.

Core GitHub Actions runs dependency-free tests on Python 3.10, 3.12 and 3.14. The optional integration test is opt-in locally; do not claim CI has run until GitHub has actually executed it.

The v0.2.0 source release already includes actual-model outputs, training manifests, main dependency pins and a report under `research/`. Independent human data review and a complete environment lockfile remain outstanding. Upload the optional adapter ZIP as a release asset if desired; keep base model weights outside this repository. Keep toy integration-test outputs separate from substantive research evidence. Use a version tag after reviewing the release.
