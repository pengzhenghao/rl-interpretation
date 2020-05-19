import copy

from ray.tune import Analysis


def remove_id(s):
    return "_".join(s.split("_")[1:])


def _read_analysis(path):
    ana = Analysis(path)
    anadfs = ana.fetch_trial_dataframes()
    tags = []
    for df in anadfs.values():
        tags.append(df.experiment_tag.unique())
    tags = [tt for t in tags for tt in t]
    return ana, anadfs, tags


def aggregate(base_result_path, new_result_path, output=None):
    # Read the data
    _, base_dfs, base_tags = _read_analysis(base_result_path)
    _, new_dfs, new_tags = _read_analysis(new_result_path)
    base_dfs = copy.deepcopy(base_dfs)
    new_dfs = copy.deepcopy(new_dfs)

    # Create helpers
    removed_tags = [remove_id(t) for t in base_tags]
    removed_tags_mapping = {
        rem_t: org_t for org_t, rem_t in
        zip(base_tags, removed_tags)
    }
    old_key_mapping = {
        rem_t: old_df_key for rem_t, old_df_key in
        zip(removed_tags, base_dfs.keys())
    }

    for df_key, val in new_dfs.items():
        key = val.experiment_tag.unique()
        assert len(key) == 1, key
        key = remove_id(key[0])

        if key in removed_tags:
            # Remove old data (it have quiet different df_key)
            print("We found {} is in original tags. original tag: {}."
                  "".format(key, removed_tags_mapping[key]))
            assert old_key_mapping[key] in base_dfs
            base_dfs.pop(old_key_mapping[key])
            new_key = removed_tags_mapping[key]
        else:
            print("We found {} is not in original tags. current id {}."
                  "".format(key, len(base_dfs)))
            new_key = "{}_{}".format(len(base_dfs), key)

        # Change the experiment name.
        val.experiment_tag = new_key
        base_dfs[df_key] = val

    if output:
        import pickle
        with open(output, "wb") as f:
            pickle.dump(base_dfs, f)
            print("Result is saved at: <{}>".format(output))

    return base_dfs
