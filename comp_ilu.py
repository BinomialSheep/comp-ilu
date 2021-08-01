# encoding=utf-8
import numpy as np
import pprint
from scipy.stats import binom
import matplotlib.pyplot as plt



# n種類のガチャをm回引いた時にコンプリートする確率
# が [m-1][n-1]に入った二次元配列
def get_comp_prob_arr(m, n):

    # m回引いた時にn種類揃ってる確率を二次元配列に入れる
    # 初期化
    result_arr = np.zeros((m, n))
    # m回引いて1種類揃う確率は(1/n)^(m-1)
    for i in range(m):
        # indexのズレと一般項の指数のズレで相殺
        result_arr[i][0] = (1/n) ** i    

    # m回引いてn種類揃う確率を漸化的に計算
    # ただし1回引いて2種類以上揃う確率は0
    for j in range(1, n):
        for i in range(1, m):
            # 直前でj種類で、被らず引く
            # 例えば直前が1種類(j=0)なら8/9。つまり(9-(j-1)+1)/9
            a = result_arr[i-1][j-1] * (n-j)/n
            # 直前でj種類で、被る
            b = result_arr[i-1][j] * (j+1)/n
            # aとbの和が求めたい確率
            result_arr[i][j] = a + b

    # for debug
    # 小数点以下3桁で出力。指数表記を禁止
    # np.set_printoptions(precision=3, suppress=True)
    # print(result_arr)

    return result_arr



# 確率pで当たるガチャをn回引いた時に当たりがm回出る確率
# が [m]に入った配列
def get_binom_prob_arr(p, n):
    binom_arr = binom.pmf(range(n+1), n, p)
    # for debug
    # np.set_printoptions(precision=3, suppress=True)
    # print(binom_arr)
    return binom_arr

# 確率pで当たるm種類のガチャをn回引いた時のコンプ率を求める巻数
def get_res_comp_prob(p, m, n):
    # binom_arr[m]は当たりがm回出る確率
    binom_arr = get_binom_prob_arr(p, n)
    # comp_arr[n][m]はn+1個当たった時にm+1種揃う確率
    comp_arr = get_comp_prob_arr(n, m)    

    # n回中i回当たりを引いて、かつその回数でコンプする確率
    result_arr = np.zeros(binom_arr.size)
    result_arr[0] = 0
    for i in range(1, binom_arr.size):
        result_arr[i] = binom_arr[i] * comp_arr[i-1][m-1]

    return np.sum(result_arr)
    


def main():
    # for debug
    # 60回引いた時のC賞（20%.全9種類）のコンプ率
    # tes = 60
    # print(f"C賞を{tes}回引いた時のコンプ率{res_comp_prob(0.2, 9, tes):.3%}")

    # 60～300回引いた時のコンプ率
    for num in range(60, 320, 20) :
        # A賞コンプ
        print(get_res_comp_prob(0.012, 2, num))
        # B賞コンプ
        print(get_res_comp_prob(0.08, 2, num))
        # C賞コンプ
        print(get_res_comp_prob(0.2, 9, num))
        # D賞コンプ
        print(get_res_comp_prob(0.15, 5, num))
        # E賞コンプ
        print(get_res_comp_prob(0.55, 10, num))

if __name__ == "__main__":
    main()