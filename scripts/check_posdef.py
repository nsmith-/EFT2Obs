import argparse
import json
import yoda
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="Rivet.yoda")
    parser.add_argument('--config', '-c', default="config_proc.json")
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--hist', type=str, default='/HiggsTemplateCrossSections/pT_Higgs')
    parser.add_argument('--rebin', type=str, default=None, help="Comma separated list of new bin edges")
    parser.add_argument('--nlo', action='store_true', help="Set if weights came from NLO reweighting")
    args = parser.parse_args()

    with open(args.config) as jsonfile:
        cfg = json.load(jsonfile)
    pars = cfg['parameters']
    defs = cfg['parameter_defaults']
    for k, v in defs.iteritems():
        for p in pars:
            p.setdefault(k, v)
    n_pars = len(pars)
    n_hists = 1 + n_pars * 2 + (n_pars * n_pars - n_pars) / 2

    infile = yoda.read(args.input)

    if args.rebin is not None:
        edges = np.array([float(X) for X in args.rebin.split(',')])
        nbins = len(edges) - 1
    else:
        edges = infile[args.hist].xEdges()
        nbins = infile[args.hist].numBins

    def getvalues(i, j):
        assert j >= i
        if i == 0 and j == 0:
            # SM point
            idx = 0
            scale = 1.0
        elif i == 0:
            # A term
            idx = 2*j - 1
            scale = pars[j-1]['val']
        elif i == j:
            # diagonal B term
            idx = 2*i
            scale = pars[i-1]['val']**2
        else:
            # cross term
            ic = n_pars*(i-1) - i*(i-1)//2 + j - i - 1
            idx = 1 + 2*n_pars + ic
            scale = pars[i-1]['val']*pars[j-1]['val']

        hname = args.hist + "[rw{:04d}{}]".format(
            idx,
            "_nlo" if args.nlo else "",
        )
        h = infile[hname]
        if args.rebin:
            h.rebinTo(edges)
        return h.areas() / scale

    hmat = np.zeros((nbins, n_pars+1, n_pars+1))
    for i in range(n_pars + 1):
        for j in range(i, n_pars + 1):
            hmat[:, i, j] = getvalues(i, j)
    hmat = (hmat + hmat.swapaxes(1, 2)) / 2
    # sumW / numEntries / (sumW_sm / numEntries)
    # numEntries cancels
    empty = hmat[:, 0, 0] == 0.0
    hmat[~empty] /= hmat[~empty, 0, 0, None, None]
    hmat[empty] = 0.0

    eig, _ = np.linalg.eigh(hmat)
    print(args.hist)
    for ibin, e in enumerate(eig):
        if e.min() >= 0:
            continue
        print("Bin {:2d} [{:4.0f}, {:4.0f}): min eigenvalue {}".format(
            ibin, edges[ibin], edges[ibin+1], e.min(),
        ))

    # Fisher at c=0 will be A@A.T times the bin constraint
    # fisher = 4*hmat[:, 1:, 0:1]*hmat[:, 0:1, 1:]
    # Per bin the vector is just A/|A|, so we can see what direction
    # without knowing the data power to constrain the bin
    for ibin in xrange(nbins):
        ev = hmat[ibin, 1:, 0]
        ev = ev / np.sqrt(np.sum(ev**2))
        top3 = abs(ev).argsort()[:-4:-1]
        top3 = "   ".join("{:5s} {:+0.2f}".format(pars[i]['name'], ev[i]) for i in top3)
        print("Bin {:2d} [{:4.0f}, {:4.0f}): fisher top constraint {}".format(
            ibin, edges[ibin], edges[ibin+1], top3
        ))

    if args.output:
        np.savez(args.output, hmat=hmat)


if __name__ == "__main__":
    main()
