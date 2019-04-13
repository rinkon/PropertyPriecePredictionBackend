import pandas as pd
from HouseRentPredictor import Regression


class HouseRentPredictionController:
    def __init__(self):
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')

    def predict(self, ms_sub_class, ms_zoning, lot_frontage, lot_area, street,lot_shape,
                land_contour, utilities, lot_config, land_slope,house_style, overall_qual,
                overall_cond, year_built, year_remod_add, exter_cond, foundation, gr_liv_area,
                full_bath, bedroom_abv_gr,kitchen_abv_gr, kitchen_qual, tot_rms_abv_grd,
                garage_cars, garage_area, garage_cond, sale_condition):

        linear_regression = Regression(None, None, None, None, None, None, self.train, self.test)
        output = linear_regression.predict(ms_sub_class, ms_zoning, lot_frontage, lot_area, street,lot_shape,
                   land_contour, utilities, lot_config, land_slope,house_style, overall_qual,
                   overall_cond, year_built, year_remod_add,exter_cond, foundation, gr_liv_area,
                   full_bath, bedroom_abv_gr,kitchen_abv_gr, kitchen_qual, tot_rms_abv_grd,
                   garage_cars, garage_area, garage_cond, sale_condition)

        return output[0]


hp = HouseRentPredictionController()
hp.predict(3, 2, 44.0, 42, 1, 3, 4, 2, 2, 2, 2, 6, 7, 1990, 1997, 2, 2, 1980, 2, 7, 22, 1, 0, 38, 8, 1.0, 2)#bad data
#hp.predict(3, 2, 44.0, 7301, 1, 3, 4, 2, 2, 2, 2, 6, 7, 2003, 2003, 2, 2, 1922, 3, 4, 22, 1, 0, 38, 8, 1.0, 2)
#hp.predict(3, 2, 64.0, 7301, 1, 3, 4, 2, 2, 2, 2, 7, 5, 2003, 2003, 2, 2, 1922, 3, 4, 1, 1, 7, 2, 672, 1.0, 2)
#hp.predict(3, 2, 44.0, 42, 1, 3, 4, 2, 2, 2, 2, 6, 7, 1990, 1997, 2, 2, 80, 92, 78, 22, 1, 0, 38, 8, 1.0, 2)#bad data


#hp.predict(160, 2, 65, 30, 2, 1, "Bnk", "ELO", "Corner", "Mod", "1Story", 10, 7, "1980-03-20", "1999-06-08", "TA", "PConc", 92, 4, 32, 60, "Fa", 60, 57, 60, "NA", "Normal")