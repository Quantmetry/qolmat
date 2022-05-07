from skopt.space import Categorical, Real, Integer


def get_search_space(tested_model, search_params):
    search_space = None
    search_name = None
    if str(type(tested_model).__name__) in search_params.keys():
        search_space = []
        search_name = []
        for name_param, vals_params in search_params[
            str(type(tested_model).__name__)
        ].items():
            search_name.append(name_param)
            if vals_params["type"] == "Integer":
                search_space.append(
                    Integer(
                        low=vals_params["min"], high=vals_params["max"], name=name_param
                    )
                )
            elif vals_params["type"] == "Real":
                search_space.append(
                    Real(
                        low=vals_params["min"], high=vals_params["max"], name=name_param
                    )
                )
            elif vals_params["type"] == "Categorical":
                search_space.append(
                    Categorical(categories=vals_params["categories"], name=name_param)
                )

    return search_space, search_name


def custom_groupby(df, groups):
    if len(groups) > 0:
        groupby = []
        for g in groups:
            groupby.append(eval("df." + g))
        return df.groupby(groupby)
    else:
        return df
