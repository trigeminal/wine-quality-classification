import lightgbm as lgb


def lightgbm_model(params):
    model = lgb.LGBMRegressor(**params)
    return model
