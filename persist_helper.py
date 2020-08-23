import numpy as np
import random
import copy
from scipy import stats

#-------------Models toxin/antitoxin system at molecular level---------------------------#

# Returns: times  --  timing of each event
#		   As     --  Number of antitoxin proteins at each time
#          Ts     --  Number of toxin proteins at each time
#          ATs    --  Number of toxin/antitoxin protein complexes at each time
#          ms     --  Number of mRNA at each time		   
#          Ps     --  Promoter state at each time
def gillespie(params, num_iters):
    A0 = params['A0']  # initial antitoxin protein conc
    T0 = params['T0']  # initial toxin protein conc
    AT0 = params['AT0']  # initial antitoxin-toxin protein complex conc
    m0 = params['m0']  # initial mRNA conc
    P0 = params['P0']  # promoter state (0 = nothing bound, 1 = antitoxin bound, 2 = AT complex bound)
    kA = params['kA']  # production rate of antitoxin protein from mRNA
    kT = params['kT']  # production rate of toxin protein from mRNA 
    km = params['km']  # production rate of mRNA when nothing is bound to the promoter
    kmA = params['kmA']  # production rate of mRNA when an antitoxin is bound to the promoter
    kmAT = params['kmAT']  # production rate of mRNA when toxin/antitoxin cprotein complex is bound to the promoter
    aA = params['aA']  # degradation of the antitoxin protein
    aT = params['aT']  # degradation of toxin protein
    aAT = params['aAT']  # degradation of antitoxin/toxin protein complex
    am = params['am']  # degradation of mRNA
    kAT = params['kAT']  # rate of formation of toxin/antitoxin protein complex
    rAT = params['rAT']  # rate of dissociation of toxin/antitoxin protein complex
    K_AT = params['K_AT']  # dissociation constant between promoter and toxin/antitoxin complex
    K_A = params['K_A']  # dissociation constant between promoter and antitoxin complex

    promoter_states = [km, kmA, kmAT]

    # arrays of molecular contents of simulation after every change
    As = [A0]
    Ts = [T0]
    ATs = [AT0]
    ms = [m0]
    Ps = [P0] # 0 = nothing bound, 1 = A is bound, 2 = AT is bound
    times = [0]

    # defines profiles of molecular changes
    profiles = [] # (A,T,AT,m)
    profiles.append([1,0,0,0])
    profiles.append([0,1,0,0])
    profiles.append([-1,0,0,0])
    profiles.append([0,-1,0,0])
    profiles.append([-1,-1,1,0])
    profiles.append([1,1,-1,0])
    profiles.append([0,0,-1,0])
    profiles.append([0,0,0,1])
    profiles.append([0,0,0,-1])

    for i in range(num_iters):

        # determines what state promoter is in
        p_A = As[i] * K_AT / (K_A * K_AT + As[i] * K_AT + ATs[i] * K_A)
        p_AT = ATs[i] * K_A / (K_A * K_AT + As[i] * K_AT + ATs[i] * K_A)
        p = random.random()
        P = 0
        if p < p_A:
            P = 1
        elif p < p_A + p_AT:
            P = 2
        Ps.append(P)

        # relative probability of each change occurring
        # (exponential)
        lambdas = []
        lambdas.append(kA * ms[i])
        lambdas.append(kT * ms[i])
        lambdas.append(aA * As[i])
        lambdas.append(aT * Ts[i])
        lambdas.append(kAT * As[i] * Ts[i])
        lambdas.append(rAT * ATs[i])
        lambdas.append(aAT * ATs[i])
        lambdas.append(promoter_states[P])
        lambdas.append(am * ms[i])

        dt = np.random.exponential(scale=1/sum(lambdas))

        ch = np.random.choice(list(range(len(lambdas))), p=lambdas / sum(lambdas))

        A,T,AT,m = np.array([As[i],Ts[i],ATs[i],ms[i]]) + np.array(profiles[ch])
        times.append(dt + times[-1])
        As.append(A)
        Ts.append(T)
        ATs.append(AT)
        ms.append(m)

    return times,As,Ts,ATs,ms,Ps

#---------Models Mutations to Toxin/Antitoxins System with Exponential State Switching--------#

# type of bacteria that uses exponential distribution for state switching
# also allows for mutations of genes
class bac1:

    def __init__(self, params, m, persist):
        self.M = params['M']  # number of possible genes
        self.m = m  # array of tuples (t_grow, t_persist)
        self.persist = persist  # array where 1 at any entry means persisting
        self.b0 = params['b0']  # birth rate when in non persistence state
        self.d0 = params['d0']  # deat rate from competition and random causes
        self.dA = params['dA']  # death rate from stressful environment
        self.mu_del = params['mu_del']  # probability of a replicated bacterium losing a given gene
        self.mu_get = params['mu_get']  # probability of a replicated bacterium obtaining a given gene
        self.mu_change = params['mu_change']  # probability of a replicated bacterium changing a given gene
        self.t_max = params['t_max']  # largest possible time scale for persistence state switching
        self.t_min = params['t_min']  # smallest possible time scale for persistence state switching
        self.params = params

    # If bacterium is in a persistence state, does not replicate
    # If bacterium is not, replicates if it would have replicated within one time step
    # according to an exponential distribution
    def replicate(self):
        if sum(self.persist) > 0:
            return None
        dt = np.random.exponential(scale=1/self.b0)
        if dt < 1:
            new_m = list(self.m)
            # possibly deletes toxin/antitoxin genes
            i = len(self.m) - 1
            while i >= 0:
                if random.random() < self.mu_del:
                    del new_m[i]
                i -= 1

            # possibly mutates toxin/antitoxin genes
            for i in range(len(new_m)):
                t_grow = new_m[i][0]
                t_persist = new_m[i][1]
                if random.random() < self.mu_change:
                    t_grow += 2 * random.random() - 1
                    t_grow = max(self.t_min, t_grow)
                if random.random() < self.mu_change:
                    t_persist += 2 * random.random() - 1
                    t_persist = max(self.t_min, t_persist)
                new_m[i] = (t_grow, t_persist)

            # possibly adds new toxin/antitoxin genes
            new_genes = np.random.binomial(self.M - len(self.m), self.mu_get)
            for i in range(new_genes):
                t_grow = random.random() * (self.t_max - self.t_min) + self.t_min
                t_persist = random.random()*(self.t_max-self.t_min) + self.t_min
                new_m.append((t_grow,t_persist))
            return bac1(self.params, new_m, [0] * len(new_m))
        return None

    # determines if bacterium enters or exits a persistence state
    def persist_change(self):
        for i in range(len(self.m)):
            if self.persist[i] == 0:
                dt = np.random.exponential(scale=self.m[i][0])
                if dt < 1:
                    self.persist[i] = 1
            else:
                dt = np.random.exponential(scale=self.m[i][1])
                if dt < 1:
                    self.persist[i] = 0
        return

    # determines if bacterium dies
    # death rate proportional to population size
    # an additional death rate is added if env != 0
    def death(self,N,env):
        if env == 0 or sum(self.persist) > 0:
            dt = np.random.exponential(scale=1/(N*self.d0))
            if dt < 1:
                return True
        else:
            dt = np.random.exponential(scale=1/(N*self.d0 + self.dA))
            if dt < 1:
                return True
        return False

# bacteria from bac1 growth (specificed in "bacs"), die, and evolve for "num_iters" time steps
# environment is stressful at times when "env_over_time" != 0
def persist_evolve1(bacs, env_over_time, num_iters):
    bac_hist = [bacs]
    for i in range(num_iters):
        new_bacs = set()
        for b in bacs:
            bac = copy.deepcopy(b)
            bac.persist_change()
            dead = bac.death(len(bacs),env_over_time[i])
            if not dead:
                new_bacs.add(bac)
                new_bac = bac.replicate()
                if new_bac:
                    new_bacs.add(new_bac)
        bac_hist.append(new_bacs)
        bacs = new_bacs
    return bac_hist

# returns (bac1) bacterial population size as a function of time from history
def get_pop_size(bac_hist):
    pops = []
    for bacs in bac_hist:
        pops.append(len(bacs))
    return pops

# returns number of persisters in (bac1) bacterial population as a function of time from history
def get_persisters(bac_hist):
    pops = []
    for bacs in bac_hist:
        num = 0
        for bac in bacs:
            if sum(bac.persist) > 0:
                num += 1
        pops.append(num)
    return pops

# returns array with number of genes each bacteria in (bac1) bacterial population has
def get_num_gene_distrib(bacs):
    num_genes = []
    for bac in bacs:
        num_genes.append(len(bac.m))
    return num_genes

# returns array with characteristic time until entering persistence 
# each bacteria in (bac1) bacterial population has
def get_growth_time_distrib(bacs):
    growth_times = []
    for bac in bacs:
        m = bac.m
        for i in range(len(m)):
            growth_times.append(m[i][0])
    return growth_times

# returns array with characteristic time until existing persistence 
# each bacteria in (bac1) bacterial population has
def get_persist_time_distrib(bacs):
    persist_times = []
    for bac in bacs:
        m = bac.m
        for i in range(len(m)):
            persist_times.append(m[i][1])
    return persist_times

#---------Models bacteria with unchanging toxin/antitoxin genes with exponential state switching------#

# bacs is a dictionary, keys are ((g1,p1,s1),(g2,p2,s2)...) which give the
# toxin/antitoxin genes and whether they are on or off
# state switching is governed by exponential distribution
def persist_test2(bacs, params, env_over_time, num_iters):
    bac_hist = [bacs]
    for i in range(num_iters):
        tot_num = 0
        for bac in bacs:
            tot_num += bacs[bac]
        new_tot = 0
        new_bacs = {}
        for typ in bacs:
            cur_num = bacs[typ]
            persister = False
            for j in range(1,len(typ)):
                if typ[j][2]:
                    persister = True
                    break
            # deals with deaths
            if env_over_time[i] == 0 or persister:
                cur_num -= np.random.binomial(bacs[typ], 1 - np.exp(-params['d0']*tot_num))
            else:
                cur_num -= np.random.binomial(bacs[typ], 1 - np.exp(-params['d0']*tot_num - params['dA']))

            # deals with births
            if cur_num > 0 and not persister:
                cur_num += np.random.binomial(cur_num, 1 - np.exp(-params['b0']))

            # deals with changes in persistence states
            temp_bacs = {typ: cur_num}
            for j in range(1,len(typ)):
                if typ[j][2]:
                    things_in_temp_bacs = list(temp_bacs.keys())
                    for typ2 in things_in_temp_bacs:
                        switchers = np.random.binomial(temp_bacs[typ2], 1 - np.exp(-1/typ[j][1]))
                        if switchers > 0:
                            typ_arr = list(typ2)
                            typ_arr[j] = (typ2[j][0], typ2[j][1], False)
                            temp_bacs[tuple(typ_arr)] = switchers
                            temp_bacs[typ2] = temp_bacs[typ2] - switchers
                else:
                    things_in_temp_bacs = list(temp_bacs.keys())
                    for typ2 in things_in_temp_bacs:
                        switchers = np.random.binomial(temp_bacs[typ2], 1 - np.exp(-1/typ[j][0]))
                        if switchers > 0:
                            typ_arr = list(typ2)
                            typ_arr[j] = (typ2[j][0], typ2[j][1], True)
                            temp_bacs[tuple(typ_arr)] = switchers
                            temp_bacs[typ2] = temp_bacs[typ2] - switchers
            for typ2 in temp_bacs:
                if typ2 in new_bacs:
                    new_bacs[typ2] += temp_bacs[typ2]
                else:
                    new_bacs[typ2] = temp_bacs[typ2]
        bac_hist.append(new_bacs)
        bacs = new_bacs
    return bac_hist

def get_pop_size2(bac_hist):
    pops = []
    for bacs in bac_hist:
        tot = 0
        for bac in bacs:
            tot += bacs[bac]
        pops.append(tot)
    return pops

def get_persisters2(bac_hist):
    persisters = []
    for bacs in bac_hist:
        tot = 0
        for bac in bacs:
            for i in range(1,len(bac)):
                if bac[i][2]:
                    tot += bacs[bac]
        persisters.append(tot)
    return persisters


#--------Models bacteria with unchanging toxin/antitoxin genes with power law state switching-----#

class bac2:
    def __init__(self, params, m, persist):
        self.m = m  # array of tuples (t_grow, t_persist)
        self.persist = persist  # array where 1 at any entry means persisting
        self.b0 = params['b0']  # birth rate when in non persistence state
        self.d0 = params['d0']  # deat rate from competition and random causes
        self.dA = params['dA']  # death rate from stressful environment
        self.params = params

        self.flip_times = self.init_flip_times() # when each persistence gene flips

    # returns when each gene will flip into or out of a persistence state
    def init_flip_times(self):
        flip_times = np.zeros(len(self.persist))
        for i in range(len(self.persist)):
            if self.persist[i] == 0:
                flip_times[i] = 1 / stats.powerlaw.rvs(self.m[i][0] + 1)
            else:
                flip_times[i] = 1 / stats.powerlaw.rvs(self.m[i][1] + 1)
        return flip_times


    # delta_t = 0 if updating persistence
    def update(self, t, delta_t, env, N):

        dead = False
        new_bac = None

        # updates persistence state
        for i in range(len(self.persist)):
            while self.flip_times[i] <= t:
                self.persist[i] = 1 - self.persist[i]
                check_birth = False
                if self.persist[i] == 0:
                    self.flip_times[i] += 1 / stats.powerlaw.rvs(self.m[i][0] + 1)
                else:
                    self.flip_times[i] += 1 / stats.powerlaw.rvs(self.m[i][1] + 1)

        # updates death
        if env == 0 or sum(self.persist) > 0:
            dt = np.random.exponential(scale=1/(N*self.d0))
            if dt < delta_t:
                dead = True
        else:
            dt = np.random.exponential(scale=1/(N*self.d0 + self.dA))
            if dt < delta_t:
                dead = True

        # checks if replicated
        if not dead and sum(self.persist) == 0:
            dt = np.random.exponential(scale=1/self.b0)
            if dt < delta_t:
                new_bac = bac2(self.params, self.m, np.zeros(len(self.m)))

        return dead,new_bac


# bacs is a dictionary, keys are ((g1,p1,s1),(g2,p2,s2)...) which give the
# toxin/antitoxin genes and whether they are on or off
# state switching is governed by power law distribution
# returns bac_hist, which is a list of tuples of pop sizes and num persisters
def persist_test_power_law(bacs, params, env_over_time, num_iters, delta_t):
    bac_hist = [(len(bacs), get_cur_persister_num(bacs))]

    for i in range(num_iters):
        new_bacs = []
        for bac in bacs:
            dead,new_bac = bac.update((i+1)*delta_t, delta_t, env_over_time[i], len(bacs))
            if not dead:
                new_bacs.append(bac)
            if new_bac != None:
                new_bacs.append(new_bac)

        bacs = new_bacs
        bac_hist.append((len(bacs), get_cur_persister_num(bacs)))

    return bac_hist


def get_cur_persister_num(bacs):
    num_persist = 0
    for bac in bacs:
        if sum(bac.persist) > 0:
            num_persist += 1
    return num_persist


def get_pop_power_law(bac_hist):
    pops = []
    for i in range(len(bac_hist)):
        pops.append(bac_hist[i][0])
    return pops

def get_persist_power_law(bac_hist):
    pops = []
    for i in range(len(bac_hist)):
        pops.append(bac_hist[i][1])
    return pops
