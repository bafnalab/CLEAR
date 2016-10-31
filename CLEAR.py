'''
Copyleft Oct 27, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
import numpy as np;
np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;
pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
import matplotlib as mpl
from numba import guvectorize,vectorize
def processSyncFileLine(x,dialellic=True):
    z = x.apply(lambda xx: pd.Series(xx.split(':'), index=['A', 'T', 'C', 'G', 'N', 'del'])).astype(float).iloc[:, :4]
    ref = x.name[-1]
    alt = z.sum().sort_values()[-2:]
    alt = alt[(alt.index != ref)].index[0]
    if dialellic:   ## Alternate allele is everthing except reference
        return pd.concat([z[ref].astype(int).rename('C'), (z.sum(1)).rename('D')], axis=1).stack()
    else:           ## Alternate allele is the allele with the most reads
        return pd.concat([z[ref].astype(int).rename('C'), (z[ref] + z[alt]).rename('D')], axis=1).stack()


def loadSync(fname = '/home/arya/workspace/CLEAR/sample_data/popoolation2/F37.sync'):
    print 'loading',fname
    cols=pd.read_csv(fname.replace('.sync','.pops'), sep='\t', header=None, comment='#').iloc[0].apply(lambda x: map(int,x.split(','))).tolist()
    data=pd.read_csv(fname, sep='\t', header=None).set_index(range(3))
    data.columns=pd.MultiIndex.from_tuples(cols)
    data.index.names= ['CHROM', 'POS', 'REF']
    data=data.sort_index().reorder_levels([1,0],axis=1).sort_index(axis=1)
    data=data.apply(processSyncFileLine,axis=1)
    data.columns.names=['REP','GEN','READ']
    data=changeCtoAlternateAndDampZeroReads(data)
    data.index=data.index.droplevel('REF')
    return data

def changeCtoAlternateAndDampZeroReads(a):
    C = a.xs('C', level=2, axis=1).sort_index().sort_index(axis=1)
    D = a.xs('D', level=2, axis=1).sort_index().sort_index(axis=1)
    C = D - C
    if (D == 0).sum().sum():
        C[D == 0] += 1
        D[D == 0] += 2
    C.columns = pd.MultiIndex.from_tuples([x + ('C',) for x in C.columns], names=C.columns.names + ['READ'])
    D.columns = pd.MultiIndex.from_tuples([x + ('D',) for x in D.columns], names=D.columns.names + ['READ'])
    return pd.concat([C, D], axis=1).sort_index(axis=1).sort_index()



def power_recursive(T, n, powers_cached):
    if n not in powers_cached.index:
        if n % 2 == 0:
            TT = power_recursive(T, n / 2,powers_cached)
            powers_cached[n]= TT.dot(TT)
        else:
            powers_cached[n]= T .dot( power_recursive(T, n - 1,powers_cached))
    return powers_cached[n]

def computePowers(T,powers):
    powers_cached =pd.Series([np.eye(T.shape[0]),T],index=[0,1])
    for n in powers:
        power_recursive(T, n, powers_cached)
    return powers_cached.loc[powers]


def precomputeTransition(data, sh,N  ):
    s,h=sh
    print  'Precomputing transitions for s={:.3f} h={:2f}'.format(s,h)
    powers=np.unique(np.concatenate(Powers(data.xs('D',level='READ',axis=1)).tolist()))
    T = Markov.computeTransition(s, N, h=h, takeLog=False)
    Tn=computePowers(T,powers)
    return Tn
Powers=lambda CD:CD.groupby(level=0,axis=1).apply(lambda x: pd.Series(x[x.name].columns).diff().values[1:].astype(int))
def HMMlik(E, Ts, CD, s):
    T = Ts.loc[s]
    powers=Powers(CD)
    likes = pd.Series(0, index=CD.index)
    for rep, df in CD.T.groupby(level=0):
        alpha = E.iloc[df.loc[(rep, 0)]].values
        for step, power in zip(range(1, df.shape[0]), powers[rep]):
            alpha = alpha.dot(T.loc[power].values) * E.values[df.loc[rep].iloc[step].values]
        likes += vectorizedLog(alpha.mean(1))
    return likes


def findML(init, gridS, cd, E, T, eps):
    i = pd.Series(True, index=init.index).values;
    mlprev = init.values.copy(True);
    mlcurrent = init.values.copy(True)
    mle = np.ones(mlcurrent.size) * gridS[0];
    ml = init.values.copy(True)
    for s in gridS[1:]:
        mlprev[i] = mlcurrent[i]
        mlcurrent[i] = HMMlik(E, T, cd[i], s)
        i = mlcurrent > mlprev + eps
        if i.sum() == 0: break
        mle[i] = s
        ml[i] = mlcurrent[i]
    return pd.DataFrame([ml, mle], index=['alt', 's'], columns=cd.index).T

def HMM(CD, E, Ts,null_s_th=None,eps=1e-1):
    print 'Fitting HMM to {} variants.'.format(CD.shape[0])
    if null_s_th is None:
        gridS=Ts.index.values
        ss=np.sort(Ts.index.values)
        null_s_th=np.abs(ss[ss!=0]).min()
    likes_null = HMMlik(E, Ts, CD, 0);
    likes_null.name = 'null'
    likes_thn = HMMlik(E, Ts, CD, -null_s_th)
    likes_thp = HMMlik(E, Ts, CD[likes_null > likes_thn], null_s_th)

    neg = likes_thn[likes_null < likes_thn]
    zero = likes_null.loc[(likes_null.loc[likes_thp.index] >= likes_thp).replace({False: None}).dropna().index];
    pos = likes_thp.loc[(likes_null.loc[likes_thp.index] < likes_thp).replace({False: None}).dropna().index];
    dfz = pd.DataFrame(zero.values, index=zero.index, columns=['alt']);
    dfz['s'] = 0
    dfp = findML(pos, gridS[gridS>=null_s_th], CD.loc[pos.index], E, Ts, eps)
    dfn = findML(neg, gridS[gridS<=-null_s_th][::-1], CD.loc[neg.index], E, Ts, eps)
    df = pd.concat([dfp, dfz, dfn])
    df = pd.concat([df, likes_null], axis=1)
    return df
import scipy.misc as sc
def getStateLikelihoods(cd, nu): c, d = cd; p = sc.comb(d, c) * (nu ** c) * (
    (1 - nu) ** (d - c)); return p
def precomputeCDandEmissions(data):
    """
    0- reads C read counts of reference  and D counts of depth
    1- computes alternate allele reads based on reference and depth
    2- saves CD
    3- saves state conditional distributions P(nu|(c,d)) aka emissions
    """
    print 'Precomputing CD (C,D)=(Derived count,total Count) and corresponding emission probabilities...'
    nu = pd.Series(np.arange(0, 1.00001, 0.001), index=np.arange(0, 1.00001, 0.001))
    c = data.xs('C', level='READ', axis=1)
    d = data.xs('D', level='READ', axis=1)
    cd = pd.concat([pd.Series(zip(c[i], d[i])) for i in c.columns], axis=1);
    cd.columns = c.columns;
    cd.index = c.index

    allreads = pd.Series(cd.values.reshape(-1)).unique();
    allreads = pd.Series(allreads, index=pd.MultiIndex.from_tuples(allreads, names=['c', 'd'])).sort_index()
    emissions = allreads.apply(lambda x: getStateLikelihoods(x, nu)).sort_index()

    index = pd.Series(range(emissions.shape[0]), emissions.index)
    CDEidx = cd.applymap(lambda x: index.loc[x])
    return emissions,CDEidx

def precomputeTransitions(data,rangeS=np.arange(-0.5,0.5001,0.1),rangeH=[0.5],N=500):
    SS,HH=np.meshgrid(np.round(rangeS,3),np.round(rangeH,3))
    SH=zip(SS.reshape(-1), HH.reshape(-1))
    return pd.Series(map(lambda sh: precomputeTransition(data,sh,N),SH),index=pd.MultiIndex.from_tuples(SH,names=['s','h'])).xs(0.5,level='h')


def Manhattan(data, columns=None, names=None, fname=None, colors=['black', 'gray'], markerSize=3, ylim=None, show=True,
              std_th=None, top_k=None, cutoff=None, common=None, Outliers=None, shade=None, fig=None, ticksize=4,
              sortedAlready=False):
    def reset_index(x):
        if x is None: return None
        if 'CHROM' not in x.columns:
            return x.reset_index()
        else:
            return x
    if type(data) == pd.Series:
        DF = pd.DataFrame(data)
    else:
        DF = data

    if columns is None: columns=DF.columns
    if names is None:names=columns

    df = reset_index(DF)
    Outliers = reset_index(Outliers)
    if not sortedAlready: df = df.sort_index()
    if not show:
        plt.ioff()
    from itertools import cycle
    def addGlobalPOSIndex(df,chroms):
        if df is not None:
            df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
            df.set_index('gpos', inplace=True);
            df.sort_index(inplace=True)

    def plotOne(b, d, name, chroms,common,shade):
        a = b.dropna()
        c = d.loc[a.index]
        if shade is not None:
            for _ ,  row in shade.iterrows():
                plt.gca().fill_between([row.gstart, row.gend], a.min(), a.max(), color='b', alpha=0.3)
        plt.scatter(a.index, a, s=markerSize, c=c, alpha=0.8, edgecolors='none')

        outliers=None
        if Outliers is not None:
            outliers=Outliers[name].dropna()
        if cutoff is not None:
            outliers = a[a >= cutoff[name]]
        elif top_k is not None:
            outliers = a.sort_values(ascending=False).iloc[:top_k]
        elif std_th is not None:
            outliers = a[a > a.mean() + std_th * a.std()]
        if outliers is not None:
            if len(outliers):
                # if name != 'Number of SNPs':
                plt.scatter(outliers.index, outliers, s=markerSize, c='r', alpha=0.8, edgecolors='none')
                    # plt.axhline(outliers.min(), color='k', ls='--')


        if common is not None:
            for ii in common.index: plt.axvline(ii,c='g',alpha=0.5)

        plt.axis('tight');
        plt.xlim(0, a.index[-1]);
        plt.ylabel(name, fontsize=ticksize * 1.5)
        # plt.title('{} SNPs, {} are red.'.format(a.dropna().shape[0], outliers.shape[0]))
        plt.xticks([x for x in chroms.mid], [str(x) for x in chroms.index], rotation=-90, fontsize=ticksize * 1.5)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.locator_params(axis='y', nbins=3)
        mpl.rc('ytick', labelsize=ticksize)
        if ylim is not None:    plt.ylim(ymin=ylim)

    chroms = pd.DataFrame(df.groupby('CHROM').POS.max().rename('len').loc[df.reset_index().CHROM.unique()] + 1000)
    chroms['offset'] = np.append([0], chroms.len.cumsum().iloc[:-1].values)
    chroms['color'] = [c for (_, c) in zip(range(chroms.shape[0]), cycle(colors))]
    chroms['mid'] = [x + y / 2 for x, y in zip(chroms.offset, chroms.len)]
    df['color'] = chroms.color.loc[df.CHROM].values
    df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
    df['color'] = chroms.color.loc[df.CHROM].values
    df.set_index('gpos', inplace=True);

    if shade is not None:
        shade['gstart']=shade.start + chroms.offset.loc[shade.CHROM].values
        shade['gend']=shade.end + chroms.offset.loc[shade.CHROM].values
    addGlobalPOSIndex(common, chroms);
    addGlobalPOSIndex(Outliers, chroms)
    if fig is None:
        fig = plt.figure(figsize=(7, 1.5*columns.size), dpi=300);
    for i in range(columns.size):
        plt.subplot(columns.size, 1, i+1);
        plotOne(df[columns[i]], df.color, names[i], chroms,common, shade)
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    plt.xlabel('Chromosome', size=ticksize * 1.5)
    if fname is not None:
        print 'saving ', fname
        plt.savefig(fname)
    if not show:
        plt.ion()
    plt.gcf().subplots_adjust(bottom=0.25)
    # sns.set_style("whitegrid", {"grid.color": "1", 'axes.linewidth': .5, "grid.linewidth": ".09"})
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': ticksize});
    mpl.rc('text', usetex=True)
    plt.show()
    return fig


class Markov:
    @staticmethod
    def computePower(T,n,takeLog=False):
        Tn=T.copy(True)
        for i in range(n-1):
            Tn=Tn.dot(T)
        if takeLog:
            return Tn.applymap(np.log)
        else:
            return Tn

    @staticmethod
    def computeTransition(s, N, h=0.5, takeLog=False):
        # s,h=-0.5,0.5;N=500
        nu0=np.arange(2*N+1)/float(2*N)
        nu_t = map(lambda x: max(min(fx(x, s, h=h), 1.), 0.), nu0)
        nu_t[0]+=1e-15;nu_t[-1]-=1e-15
        return pd.DataFrame(computeTransition(nu_t),index=nu0,columns=nu0)


    @staticmethod
    def computeProb(X,T):
        return sum([np.log(T.loc[X[t,r],X[t+1,r]]) for t in range(X.shape[0]-1) for r in range(X.shape[1])])

def fx(x, s=0.0, h=0.5): return ((1 + s) * x ** 2 + (1 + h * s) * x * (1 - x)) / (
(1 + s) * x ** 2 + 2 * (1 + h * s) * x * (1 - x) + (1 - x) ** 2)






@guvectorize(['void(float64[:], float64[:,:])'],'(n)->(n,n)')
def computeTransition(nu_t,T):
    N = (nu_t.shape[0] - 1) / 2
    logrange= np.log(np.arange(1,2*N+1))
    lograngesum=logrange.sum()
    lognu_t=np.log(nu_t)
    lognu_tbar=np.log(1-nu_t)
    for i in range(T.shape[0]):
        for j in range(T.shape[0]):
            T[i,j]= np.exp(lograngesum - logrange[:j].sum() - logrange[:2*N-j].sum()+ lognu_t[i]*j +  lognu_tbar[i]*(2.*N-j))
        if not nu_t[i]: T[i, 0] = 1;
        if nu_t[i] == 1: T[i, -1] = 1;
@vectorize
def vectorizedLog(x):
    return float(np.log(x))


def scanGenome(genome, f, winSize=50000, step=10000, minVariants=None):
    """
    Args:
        genome: scans genome, a series which CHROM and POS are its indices
        windowSize:
        step:
    Returns:

    """
    res=[]
    if minVariants is not None: f.update({'COUNT': lambda x: x.size})
    for chrname,chrom in genome.groupby(level='CHROM'):
        df=pd.DataFrame(scanChromosome(chrom,f,winSize,step))
        df['CHROM']=chrname;df.set_index('CHROM', append=True, inplace=True);df.index=df.index.swaplevel(0, 1)
        res+=[df]
    df = pd.concat(res)
    if minVariants is not None:
        df[df.COUNT < minVariants] = None
        df = df.loc[:, df.columns != 'COUNT'].dropna()
    return df


def scanChromosome(x,f,winSize,step):
    """
    Args:
        chrom: dataframe containing chromosome, positions are index and the index name should be set
        windowSize: winsize
        step: steps in sliding widnow
        f: is a function or dict of fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
    Returns:
    """
    POS=x.index.get_level_values('POS')
    res=[]
    def roundto(x, base=50000):return int(base * np.round(float(x)/base))
    Bins=np.arange(max(0,roundto(POS.min()-winSize,base=step)), roundto(POS.max(),base=step),winSize)
    for i in range(int(winSize/step)):
        bins=i*step +Bins
        windows=pd.cut( POS, bins,labels=(bins[:-1] + winSize/2).astype(int))
        res+=[x.groupby(windows).agg(f)]
        res[-1].index= res[-1].index.astype(int);res[-1].index.name='POS'
    return pd.concat(res).sort_index().dropna()

