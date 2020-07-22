import pandas as pd
import numpy as np


# takes a row of a box score df
# returns the estimated number of possessions (per team) in the game
# uses the possession estimation formula documented at https://www.basketball-reference.com/about/glossary.html#poss
# formula is symmetrical so no point in calculating away/home possessions separately
def EstimatePossessions(row):
    awayPoss = 0.5 *(
        (row["AwayFGA"] + 0.4 * row["AwayFTA"] - 1.07 * (row["AwayORB"] / (row["AwayORB"] + row["HomeDRB"])) * 
             (row["AwayFGA"] - row["AwayFGM"]) + row["AwayTOV"])
        +
        (row["HomeFGA"] + 0.4 * row["HomeFTA"] - 1.07 * (row["HomeORB"] / (row["HomeORB"] + row["AwayDRB"])) *
             (row["HomeFGA"] - row["HomeFGM"]) + row["HomeTOV"])
                    )
    return awayPoss


# TODO: adapt for NBA data
class EnhancedSRS():
    
    def __init__(self):
        self.teams = None
        self.ratings = None
        
    def fit(self, X):
        self.teams = list(np.sort(X.AwayTeam.unique()))
        
        # get total (pace-normalized) point differentials for the season and number of games played
        self.totalDiffs = {}
        self.totalGames ={}
        for t in self.teams:
            aGames = X.loc[X.AwayTeam == t]
            hGames = X.loc[X.HomeTeam == t]
            self.totalDiffs[t] = aGames.AdjDiff.sum() - hGames.AdjDiff.sum()
            self.totalGames[t] = len(aGames) + len(hGames)
        self.totalDiffs = pd.Series(self.totalDiffs).sort_index()
        self.totalGames = pd.Series(self.totalGames).sort_index()
        
        # linear algebra to solve for team ratings
        
        # construct matrix of matchups for least squares equation
        A = np.zeros((len(self.teams), len(self.teams)))
        b = -1 * self.totalDiffs.values
        for i, r in X.iterrows():
            away = r["AwayTeam"]
            home = r["HomeTeam"]
            
            A[self.teams.index(home), self.teams.index(away)] += 1
            A[self.teams.index(away), self.teams.index(home)] += 1
        
        # rearrange matrix/vector equation into Ax = b form
        for i in range(len(A)):
            A[i, i] = -1 * self.totalGames.iat[i]
        
        # solve the least squares equation
        self.ratings = np.linalg.lstsq(A, b, rcond=None)[0]
        self.ratings = pd.Series(self.ratings, index=self.teams)
    
    # 1 == home wins, 0 == away wins
    def predict(self, X):
        preds = np.zeros(len(X))
        
        p = 0
        for i, row in X.iterrows():
            away = row["AwayTeam"]
            home = row["HomeTeam"]
            
            if self.ratings[home] > self.ratings[away]:
                preds[p] = 1
            p += 1
        
        return preds
    
    def score(self, X):
        preds = self.predict(X)
        
        true = np.zeros(len(X))
        t = 0
        for i, row in X.iterrows():
            if row["ScoreDiff"] < 0:
                true[t] = 1
            t += 1
        
        return 1 - (np.sum(abs(preds - true)) / len(true))


# TODO: refactor functions to use (X, y) syntax
# TODO: predict spread
# TODO: use linear? logistic? models for predictions
# TODO: predict probabilities
class WinPct():
    
    def __init__(self):
        self.teams = None
        self.pcts = None
    
    def fit(self, X):
        self.teams = list(np.sort(X.V.unique()))
        self.pcts = pd.Series(index=self.teams)
        
        for t in self.teams:
            aGames = X.loc[X.V == t]
            hGames = X.loc[X.H == t]
            wins = len(aGames.loc[X.Result > 0]) + len(hGames.loc[X.Result < 0])
            total = len(aGames) + len(hGames)
            self.pcts[t] = wins / total
    
    def predictSU(self, X):
        preds = np.zeros(len(X))
        
        p = 0
        for i, row in X.iterrows():
            away = row["V"]
            home = row["H"]
            
            if home not in self.pcts or away not in self.pcts:
                preds[p] = np.nan
            elif self.pcts[home] > self.pcts[away]:
                preds[p] = 1
            p += 1
        
        return preds
    
    def score(self, X):
        preds = self.predictSU(X)
        
        true = np.zeros(len(X))
        t = 0
        for i, row in X.iterrows():
            if row["Result"] < 0:
                true[t] = 1
            t += 1
        
        return 1 - (np.sum(abs(preds - true)) / len(true))
