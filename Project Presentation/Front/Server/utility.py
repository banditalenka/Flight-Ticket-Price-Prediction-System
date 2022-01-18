import json
import pickle
import numpy as np

__model = None
__location = None
__data_columns = None
__flight = None
__source = None
__destination = None


def get_estimated_price(Total_stops, Date_Diff,Journey_day, Journey_month,Booking_Day,Booking_Month,Dep_hour,Dep_min,dur_hour,
                        dur_min, Air_India, GoAir, IndiGo, Jet_Airways, Jet_Airways_Business, Multiple_carriers,
                        Multiple_carriers_Premium_economy, SpiceJet, Trujet, Vistara, Vistara_Premium_economy,
                        s_Chennai, s_Delhi, s_Kolkata, s_Mumbai, d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata,
                        d_New_Delhi):
    load_saved_artifacts()
    print("PREDICTION")
    prediction = __model.predict([[
        Total_stops, Date_Diff, Journey_day, Journey_month, Booking_Day, Booking_Month, Dep_hour, Dep_min, dur_hour,
        dur_min, Air_India, GoAir, IndiGo, Jet_Airways, Jet_Airways_Business, Multiple_carriers,
        Multiple_carriers_Premium_economy, SpiceJet, Trujet, Vistara, Vistara_Premium_economy,
        s_Chennai, s_Delhi, s_Kolkata, s_Mumbai, d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata,
        d_New_Delhi
    ]])

    output = round(prediction[0], 2)
    print(output)
    return output


def get_location_names():
    load_saved_artifacts()
    return __location

def get_flight_names():
    load_saved_artifacts()
    b=[]
    for i in __flight:
        b+=i.split("_")
    for i in b:
        if "airline" in i:
            b.remove("airline")
    b = [i.title() for i in b]
    return b

def get_source_names():
    load_saved_artifacts()
    b=[]
    for i in __source:
        b+=i.split("_")
    for i in b:
        if "source" in i:
            b.remove("source")
    b = [i.title() for i in b]
    return b

def get_destination_names():
    load_saved_artifacts()
    b=[]
    for i in __destination:
        b+=i.split("_")
    for i in b:
        if "destination" in i:
            b.remove("destination")
    b = [i.title() for i in b]
    return b



def load_saved_artifacts():
    global __model
    global __location
    global __data_columns
    global __flight
    global __source
    global __destination

    with open("./artifacts/columns_diff.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[0:]
        __flight = __data_columns[10:21]
        __source = __data_columns[21:25]
        __destination = __data_columns[25:]

    with open("./artifacts/Flight_rf_diff.pickle", 'rb') as f:
        print("Load FILE")
        __model = pickle.load(f)


if __name__ == '__main__':
    print(get_location_names())
    print(get_flight_names())
    print(get_source_names())
    print(get_destination_names())
