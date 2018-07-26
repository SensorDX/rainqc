from model.hurdle_regression import MixLinearModel
import pandas as pd
import numpy as np

def main():
    """
    Input:
        target_station: target station
        datetime: datetime for qc.

        process:
          -
    :return:
    """

    df = pd.read_csv('sampletahmo.csv')
    w = np.random.rand(100, 6)
    #print df.head(5)
    y, x = w[:, 0], w[:, 1:]
    # print y
    y,x = df.iloc[:,1].as_matrix(), df.iloc[:,2:].as_matrix()

    mixl = MixLinearModel()
    mixl.reg_model.fit(x, y)
    yz = (y > 0.5).astype(int)
    mixl.log_reg.fit(x, yz)
    res = y.reshape(-1, 1) - mixl.reg_model.predict(x).reshape(-1, 1)
    # print res, res.shape
    # mixl.kde.fit(res)
    # print mixl.kde.score_samples(y.reshape(-1,1))
    mixl.train(y, x)
    #print mixl.predict(y, x)
    #mixl.save()
    mxx = MixLinearModel()
    mxx.to_json()
    mxx.from_json()
    print mxx.predict(y, x)


if __name__ == '__main__':
    main()



