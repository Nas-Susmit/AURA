# GitHub release and mentor/reviewer checklist

The ZIP is a complete repository tree, not a hosted repository. No account login, public repository creation or push was performed.

1. Confirm authorship, supervisor consent, project permissions, university policy and the venue's anonymous-review/preprint rules. If double blind, remove identifying metadata, author names, manuscript author block, notebook byline, and identity-bearing Git commit history before sharing an anonymized version. Do not send an identifying GitHub link to anonymous reviewers.
2. Decide on permissions. MIT or BSD-3-Clause is common for original software, but obtain author approval and determine separate manuscript/data rights first. The package deliberately contains no license grant.
3. On GitHub, create an empty repository called `AURA` (private initially). Do not initialize with README, .gitignore or license if using the following commands. Extract ZIP and enter the `AURA` directory.

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Release AURA V3 reproducible simulation benchmark"
git remote add origin https://github.com/YOUR_USERNAME/AURA.git
git push -u origin main
```

Replace `YOUR_USERNAME` with the actual GitHub account. Authenticate through GitHub CLI or Git Credential Manager if prompted; never put a token in the repository or command history. If repository already exists, clone first and copy files into it instead of force-pushing.

4. Check Actions tab, run workflow **AURA checks** manually to reproduce the full benchmark. CI status is not certified until it has run on GitHub. Compare the output with `docs/reproduction_report.txt`.
5. When allowed, tag the reviewed code `v3.0.0` and create a release. Record the commit SHA/tag in the manuscript Data and Code Availability section, replace the local archive name/link with the actual GitHub URL, and recompile. A release archive or Zenodo DOI can be useful once published; no DOI is claimed here.
6. Verify that repository contents, figures and notebook disclose synthetic-only scope and AI assistance. Ensure no secrets or private medical data are present. Make public only when approved, or use the conference's approved anonymous supplement route.

For mentor review, use `docs/mentor_demo.html` for a quick no-install walkthrough and `AURA_Mentor_Demo.ipynb` for live execution. Then inspect the primary disagreement contrast, strong logistic control and saved evidence. Ask the mentor to independently run the full benchmark and audit the methodological assumptions.
