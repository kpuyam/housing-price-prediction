import pandas as pd
import unittest

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ta_lib.attribs_adder import FeatureExtraction


class TestFeatureExtraction(unittest.TestCase):

    def test_transform(self):
        # Load the dataset
        data = pd.read_parquet("data/train/housing/features.parquet")
        
        # Number of columns before transformation
        num_cols_before = data.shape[1]
        
        # Define numerical and categorical attributes
        num_attribs = list(data.drop("ocean_proximity", axis=1))
        cat_attribs = ["ocean_proximity"]
        
        # Define numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', FeatureExtraction()),
            ('std_scaler', StandardScaler()),
        ])
        
        # Define features transformer
        features_transformer = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        
        # Fit and transform the dataset
        transformed_data = features_transformer.fit_transform(data)
        
        # Number of columns after transformation
        num_cols_after = transformed_data.shape[1]
        
        # Check if the number of columns increases by 7 after transformation
        self.assertEqual(num_cols_after, num_cols_before + 7)

if __name__ == "__main__":
    unittest.main()
