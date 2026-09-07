# Chrome verification inventory — focused quality fixes

Use an isolated local app server and ordinary Google Chrome controls. Do not interrupt the existing main app or its training jobs. Test media, datasets, voices, and outputs have QA-specific names and directories.

| Claim / interaction | Functional check | Visible evidence |
|---|---|---|
| Oversized English clauses preserve words | Generate the previously failing long sentence with normal English defaults; inspect emitted segments and completed audio | Completed generation, playable audio, log segments preserving `across` |
| Selected language controls numeric normalization | Generate `123` with English selected; repeat a number separated by an explicit pause | Completed English audio and normalized-text log |
| Contractions preserve meaning | Generate `She's been waiting for you. It's already been completed.` | Completed playable audio and unchanged contraction text in preprocessing |
| Caption language and content are matched | Prepare a short source with two language-tagged sidecars; inspect selected language and generated manifest | Preparation completion/report and selected sidecar |
| Plain TXT is aligned to spoken words | Prepare a short video + TXT containing written numbers, with no subtitle file | Correct clip transcript boundaries in report/manifest and playable clips |
| Weak/mismatched captions cannot become trusted training text | Prepare a source with deliberately wrong captions | Specific rejected/unverified explanation; no misleading successful dataset of wrong pairs |
| Incomplete segments are recovered | Generate a passage with a reduced acoustic-token limit, exercising automatic split/retry | Retry progress followed by completed playable full output |
| Recovery has a finite failure mode | Use an impractically small generation limit | Clear incomplete-speech failure and no falsely completed audio |
| FP32 full-module checkbox defaults to on | Open training from a fresh QA UI state | Checkbox visibly enabled and explanatory help |
| Checkbox can select either storage precision | Toggle off, save/apply configuration, toggle on again; launch short isolated training checks when GPU is available | Saved configuration matches both states; training log reports actual storage dtype |
| Existing workflows remain usable | Normal generation, preparation, and training initiation through their regular controls | No browser console/server errors caused by changed paths |

Inspect the initial viewport and the changed training section visually. Check an ordinary desktop window and a smaller desktop viewport, including error/progress states. Perform a brief exploratory pass covering language changes, empty input, and returning the precision checkbox to its enabled default. Record actual outcomes and any limitations in the final QA report.
