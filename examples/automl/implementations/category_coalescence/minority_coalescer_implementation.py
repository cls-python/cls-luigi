import numpy as np

class MinorityCoalescer(object):
    def __init__(self, minimum_fraction: float = None) -> None: 
        self.minimum_fraction = minimum_fraction

        

    
    def fit(self, X, y=None) -> 'MinorityCoalescer':
            
        if self.minimum_fraction is None:
            return self
        
        do_not_coalesce = list()
        for column in range(X.shape[1]):
            do_not_coalesce.append(set())

            unique, counts = np.unique(X[:, column], return_counts=True)
            colsize = X.shape[0]

            for unique_value, count in zip(unique, counts):
                fraction = float(count) / colsize
                if fraction >= self.minimum_fraction:
                    do_not_coalesce[-1].add(unique_value)
        
        self.do_not_coalesce_ = do_not_coalesce
        return self
    
    def transform(self, X):
        if self.minimum_fraction is None:
            return self
        

        for column in range(X.shape[1]):
            dtype = type(X[:, column][0])
            unique = np.unique(X[:, column])
            unique_values = [
                unique_value
                for unique_value in unique
                if unique_value not in self.do_not_coalesce_[column]
            ]
            mask = np.isin(X[:, column], unique_values)

            if dtype == str:
                X[mask, column] = "1"
            else:
                X[mask, column] = 1

        return X
    


if __name__ == "__main__":

    import pandas as pd

    cs = MinorityCoalescer(minimum_fraction=0.2)
    df = pd.DataFrame()
    cats = []

    for i in range(0, 100):
        cats.append(1)


    for i in range(0, 51):
        cats.append(2)



    for i in range(0, 26):
        cats.append(3)


    for i in range(0,2):
        cats.append(4)



    df["cats"] = cats
    df["cats"] = df["cats"].astype("category")

    df["cats2"] = cats
    df["cats2"] = df["cats2"].astype("category")

    df["num"] = 1
    df["str"] = "a"


    cat_df = df[df.select_dtypes(include=['category']).columns.tolist()]


    cs.fit(cat_df.values)
    coalesced = cs.transform(cat_df.values)

    print(coalesced)

    df[df.select_dtypes(include=['category']).columns.tolist()] = coalesced

   






    




        


    

