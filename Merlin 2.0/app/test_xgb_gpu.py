import numpy as np, xgboost as xgb, time
n = 20000
X = np.random.randn(n, 50).astype("float32")
y = (np.random.rand(n) > 0.5).astype("int32")
dtrain = xgb.DMatrix(X, label=y)

params = dict(objective="binary:logistic",
              tree_method="gpu_hist",   # <— requires GPU build
              predictor="gpu_predictor")

t0 = time.time()
bst = xgb.train(params, dtrain, num_boost_round=200)
print("Trained OK in %.2fs" % (time.time()-t0))
print("Attrs:", bst.attributes())
