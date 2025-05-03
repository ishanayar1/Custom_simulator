
import numpy as np
from datetime import datetime
import random
import math
from object_creation_modified import Junction, Trains, obj_creation, arranging_sections,find_junction_by_name_line,find_section_by_name,find_line_by_number,find_common_junction

class Passenger:
    def __init__(self, name, line, arrival_time, boarding_station, destination):
        self.name = name
        self.line = line
        self.arrival_time = arrival_time
        self.boarding_station = boarding_station
        self.destination = destination
        self.leaving_time = ""
        self.delay = 0
        self.total_time = 0
        self.shift = 0
        

def find_junction_by_name_line(junction_list, target_name, target_line):
    for junction in junction_list:
        if junction.name == target_name and abs(junction.line) == abs(target_line):
            return junction
    return None

def passenger_parse(file_path):
    passenger_list = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, line_num, arrival_time, boarding_station, destination = line.split(",")
            passenger = Passenger(name, int(line_num), arrival_time, boarding_station, destination)
            passenger_list.append(passenger)
    return passenger_list
     
def passenger_arrival(passenger_list, junction_list):
    for passenger in passenger_list:
        boarding_junction = find_junction_by_name_line(junction_list, passenger.boarding_station, passenger.line)
        
        destination_junction = None
        for junc in junction_list:
            if junc.name == passenger.destination and junc.signal == 1:
                destination_junction = junc
                break
        
        if destination_junction is None:
            print(f"Destination junction {passenger.destination} does not exist or is inactive.")
        if boarding_junction is None:
            print(f"Boarding junction {passenger.boarding_station} does not exist.")
        else:
            boarding_junction.passengerqueue.append((passenger.arrival_time, passenger))
    
    return passenger_list, junction_list


def time_string(time: str, mins: int) -> str:
    hrs = int(time[:2])
    minutes = int(time[3:])

    minutes += mins

    hrs += minutes // 60
    minutes %= 60

    hr_string = f"{hrs:02}"
    min_string = f"{minutes:02}"

    return f"{hr_string}:{min_string}"

def next_stations(station, line_list):
    destination = []
    curr_station = False

    for line in line_list:
        if line.line_number == station.line:
            junctions = line.junctions
            for junction in junctions:
                if junction == station:
                    curr_station = True
                    continue
                if curr_station and junction.signal == 1:
                    destination.append(junction)
            break
    return destination

def generate_passenger_arrivals(start_time, num_passengers, lambda_rate):
    num_passengers = int(num_passengers)

    inter_arrival_times = np.random.exponential(1/lambda_rate, num_passengers)
    sum = np.cumsum(inter_arrival_times)

    arrival_times = []
    current_time = start_time
    for i in sum:
        current_time = time_string(start_time, int(i))
        arrival_times.append(current_time)

    return arrival_times

def generate_passenger_data(passenger_data, junction, lambda_value, num_passengers, line_list, junction_list, output_file):
    start_time = "08:00"
    current_lambda = lambda_value
    arrival_times = generate_passenger_arrivals(start_time, num_passengers, current_lambda)

    valid_junctions = [j for j in junction_list if isinstance(j, Junction)]
    common_junction_map = find_common_junction(valid_junctions)

    destination_candidates = [
        j for j in next_stations(junction, line_list)
        if j.signal == 1 and j != junction
    ]

    if junction not in common_junction_map:

        next_junctions = next_stations(junction, line_list)
        transfer_junctions_ahead = [
            j for j in next_junctions
            if j in common_junction_map and j.line == junction.line
        ]

        for transfer_junc in transfer_junctions_ahead:
            connected_junctions = common_junction_map[transfer_junc]
            for connected_junc in connected_junctions:
                if connected_junc.line != junction.line:
                    new_destinations = next_stations(connected_junc, line_list)
                    destination_candidates.extend([
                        j for j in new_destinations
                        if j.signal == 1 and j != junction
                    ])

    destination_candidates = list({j.name: j for j in destination_candidates}.values())

    print(f'For junction {junction.name} possible destinations are:')
    for j in destination_candidates:
        print(f'  {j.name}')

    for arrival_time in arrival_times:
        if destination_candidates:
            pass_destination = random.choice(destination_candidates)
            passenger = {
                'name': f"P{len(passenger_data)+1}",
                'line': junction.line,
                'arrival_time': arrival_time,
                'boarding_station': junction.name,
                'destination': pass_destination.name
            }
            passenger_data.append(passenger)

    return passenger_data


def total_time(arrival,departure):
    time_format = "%H:%M"
    arrival_time = datetime.strptime(arrival, time_format)
    departure_time = datetime.strptime(departure, time_format)
    time_difference = departure_time - arrival_time
    diff = time_difference.total_seconds() / 60
    return diff

def distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def get_distance_between_junction(boarding_junction, destination_junction):

     x1= boarding_junction.x
     x2= destination_junction.x
     y1= boarding_junction.y
     y2= destination_junction.y
     distance_travelled = distance(x1,y1,x2,y2)

     return distance_travelled
