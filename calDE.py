import argparse


# c10_de = [
#     0.3382,
#     0.2489,
#     0.2252,
#     0.2091,
#     0.2005,
#     0.1965
# ] # c10
c10_de =[
    1.8399,
    1.6219,
    1.5461,
    1.5096,
    1.4802
] # t200
# c10_de =[
#     1.4745,
#     1.2746,
#     1.2066
# ] # vi1000
# c10_de =[
#     1.3960,
#     1.2542,
#     1.1969
# ] # vi1000 (calibrated)


def main(config):
    nll = config.nll
    prev_nll = None
    dee = 0
    for i, de_nll in enumerate(c10_de[::-1]):
        m = len(c10_de) - i
        if de_nll > nll and prev_nll is None:
            denom = c10_de[-2]-c10_de[-1]
            nomin = m*(c10_de[-2]-nll)-(m-1)*(c10_de[-1]-nll)
            dee = nomin/denom
            break
        if de_nll > nll:
            dee = m + (de_nll-nll)/(de_nll-prev_nll)
            break
        prev_nll = de_nll
    print(f"DEE {dee}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nll", default=None, type=float)
    args = parser.parse_args()

    main(args)
