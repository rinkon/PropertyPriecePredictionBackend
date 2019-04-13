import flask
import operations
from flask import request
from flask import jsonify
from flask_cors import CORS
import HouseRentPredictionController

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)
hp = HouseRentPredictionController.HouseRentPredictionController()

@app.route('/', methods=['GET'])
def home():
    return ""


@app.route('/predict', methods=['GET', 'POST'])
def search():
    # val1 = request.args.get("x")
    # val2 = request.args.get("y")
    content = request.get_json(silent=True)
    ms_sub_class = int(content["MSSubClass"])
    ms_zoning = int(content["MSZoning"])
    lot_frontage = int(content["LotFrontage"])
    lot_area = int(content["LotArea"])
    street = int(content["Street"])
    lot_shape = int(content["LotShape"])
    land_contour = int(content["LandContour"])
    utilities = int(content["Utilities"])
    lot_config = int(content["LotConfig"])
    land_slope = int(content["LandSlope"])
    house_style = int(content["HouseStyle"])
    overall_qual = int(content["OverallQual"])
    overall_cond = int(content["OverallCond"])
    year_built = 1990 #int(content["YearBuilt"])
    year_remod_add = 1997 #int(content["YearRemodAdd"])
    exter_cond = int(content["ExterCond"])
    foundation = int(content["Foundation"])
    gr_liv_area = int(content["GrLivArea"])
    full_bath = int(content["FullBath"])
    bedroom_abv_gr = int(content["BedroomAbvGr"])
    kitchen_abv_gr = int(content["KitchenAbvGr"])
    kitchen_qual = int(content["KitchenQual"])
    tot_rms_abv_grd = int(content["TotRmsAbvGrd"])
    garage_cars = int(content["GarageCars"])
    garage_area = int(content["GarageArea"])
    garage_cond = int(content["GarageCond"])
    sale_condition = int(content["SaleCondition"])

    sell_price = hp.predict(ms_sub_class, ms_zoning, lot_frontage, lot_area, street, lot_shape,
                land_contour, utilities, lot_config, land_slope, house_style, overall_qual,
                overall_cond, year_built, year_remod_add, exter_cond, foundation, gr_liv_area,
                full_bath, bedroom_abv_gr, kitchen_abv_gr, kitchen_qual, tot_rms_abv_grd,
                garage_cars, garage_area, garage_cond, sale_condition)
    return jsonify(sell_price=sell_price)


app.run()


