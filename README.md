# MLPerf™ Inference v1.0

## GitHub Submission HOWTO

### Clone the MLPerf Inference v1.0 submission tree

Clone the submission tree e.g. under your home directory:

```bash
$ export SUBMISSION_ROOT=$HOME/submissions_inference_1_0
$ git clone git@github.com:mlcommons/submissions_inference_1_0.git $SUBMISSION_ROOT
$ cd $SUBMISSION_ROOT
```

### Create a branch

We recommend creating a new branch for every logically connected group of
results e.g. all results from your System-Under-Test (SUT) or only relating to
a particular benchmark. Prefix your branch name with your organization's name.
Feel free to include the SUT name, implementation name, benchmark name, etc.

For example:

```bash
$ git checkout master && git pull
$ git checkout -b dividiti-closed-aws-g4dn.4xlarge-openvino
```

Populate your branch according to the [submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#directory-structure).

You can inspect your changes:

```bash
$ git status
On branch dividiti-closed-aws-g4dn.4xlarge-openvino
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        closed/dividiti/code/
        closed/dividiti/compliance/
        closed/dividiti/measurements/
        closed/dividiti/results/
        closed/dividiti/systems/

nothing added to commit but untracked files present (use "git add" to track)
```

and make intermediate commits as usual:

```bash
$ git add closed/dividiti
$ git commit -m "Dump repo:mlperf-closed-aws-g4dn.4xlarge-openvino."
```

### Run the submission checker

Once you are happy with the tree structure, [truncate the accuracy logs](https://github.com/mlcommons/inference/blob/master/tools/submission/truncate_accuracy_log.py) and [run the submission checker](https://github.com/mlcommons/inference/blob/master/tools/submission/submission-checker.py), culminating in e.g.:

```
      INFO:main:Results=2, NoResults=0
      INFO:main:SUMMARY: submission looks OK
```

### Push the changes

Once you **and the submission checker** are happy with the tree structure, you can push the changes:

```bash
$ git push

fatal: The current branch dividiti-closed-aws-g4dn.4xlarge-openvino has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin dividiti-closed-aws-g4dn.4xlarge-openvino
```

Do exactly as suggested:

```bash
$ git push --set-upstream origin dividiti-closed-aws-g4dn.4xlarge-openvino
```

### Create a pull request

If you now go to https://github.com/mlcommons/submissions_inference_1_0/, you should see a notification
about your branch being recently pushed and can immediately create a pull request (PR).
You can also select your branch from the dropdown menu under `<> Code`. (Aren't you happy you prefixed your branch's name with the submitter's name?)

As usual, you can continue committing to the branch until the PR is merged, with any changes
being reflected in the PR.
