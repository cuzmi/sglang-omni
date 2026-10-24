# SGLang-Omni Results

A shared home for SGLang-Omni benchmark, accuracy and profiling results.
Results are organized by the model or component being studied, with dated
experiment directories inside each topic.

## Results index

| Topic | Results |
| --- | --- |
| MiniCPM-o Flow/HiFT | [Serving performance, accuracy and overlap measurements](minicpm-flow-hift/README.md) |

## Adding results

- Use a short, descriptive topic directory, such as `minicpm-flow-hift`.
- Keep each experiment in a dated directory and link its report from the topic README.
- Add new topics to the index above; a PR number is not required.
- Include reports, measurement data, logs, figures and the environment/source versions needed to interpret them. Preserve limitations and unsuccessful results.
- Keep implementation code, reproduction scripts, patches and source bundles in their source repositories or archives, referenced by fixed-commit links.
- Update `SHA256SUMS.json` when published files change. It covers this branch's files and excludes itself.
