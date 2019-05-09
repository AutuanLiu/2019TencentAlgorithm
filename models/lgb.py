import lightgbm as lgb

class LGB_Module:
    def __init__(self, train_data, valid_data, params):
        self.train_data = lgb.Dataset(*train_data)
        self.valid_data = lgb.Dataset(*valid_data, reference=lgb_train)
        self.params = params
    
    def train_valid():
        self.gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_valid,early_stopping_rounds=5
        self.gbm.save_model('model.txt')
        y_pred = self.gbm.predict(self.valid_data[0])
        score = mean_squared_error(self.valid_data[1], y_pred) ** 0.5
        return score
    
    def test(test_data):
        y = self.gbm.predict(test_data)
        return y