import pickle


def price_prediction(features):
    pickled_model = pickle.load(open('House_model.pkl', 'rb'))
    price = str(round(list(pickled_model.predict([features]))[0]))

    return str("House price may be " + price)
test_features=[3.0, 2.25, 1480.0, 5400.0, 2.0, 0.0, 1480.0, 0.0, 1914.0, 2014.0]
price_prediction(test_features)