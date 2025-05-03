from time import time
from datetime import datetime
import math
import random
import numpy as np
from passenger_generation import Passenger, passenger_parse, passenger_arrival, generate_passenger_data, total_time, get_distance_between_junction
from object_creation_modified import Junction, Trains, obj_creation, arranging_sections,find_junction_by_name_line,find_section_by_name,find_line_by_number,find_junction_by_name,find_common_junction,arrange_junction
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from DQN_Agent import DQNAgent, get_state 

class timetable:
    def __init__(self, train, section, entering_time, leaving_time):
        self.train = train
        self.section = section
        self.entering_time = entering_time
        self.leaving_time = leaving_time

    def print(self):
        print(f"Train: {self.train}, Section: {self.section}, Entering Time: {self.entering_time}, Leaving Time: {self.leaving_time}")

def distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def time_string(time: str, mins: int) -> str:
    hrs = int(time[:2])
    minutes = int(time[3:])

    minutes += mins

    hrs += minutes // 60
    minutes %= 60

    hr_string = f"{hrs:02}"
    min_string = f"{minutes:02}"

    return f"{hr_string}:{min_string}"

def find_passenger_by_name(passenger_list, target_name):
    for passenger in passenger_list:
        if passenger.name == target_name:
            return passenger
    return None

def initial_signaling(junction_list):
    junc_dict = {}
    for i in junction_list:
        if i.signal == 1:
            if i.name not in junc_dict:
                junc_dict[i.name] = []
            junc_dict[i.name].append(i)
            
    for i in junc_dict:
        for j in range(len(junc_dict[i])):
            jun = junc_dict[i][j]
            if j == 0:
                jun.set_status(0, time_string("08:00", 5))
            else:
                jun.set_status(1, time_string("08:00", 5*(len(junc_dict[i]) - 1)))
                
def signaling_1(junction_list, junction):
    new_status = 0
    new_time = ""
    junc_list = []
    current_status = junction.get_status()
    current_time = junction.get_event()
    for i in junction_list:
        if junction.name == i.name and junction.signal == 1:
            junc_list.append(i)
    if len(junc_list) == 1:
        new_time = time_string(current_time, 5)
        if current_status == 0:
            new_status = 1
        else:
            new_status = 0
    else:
        if current_status == 1:
            new_status = 0
            new_time = time_string(current_time, 5)
        else:
            new_status = 1
            new_time = time_string(current_time, 5*(len(junc_list) - 1))
    return new_status,new_time

def signaling(junction_list, junction):
    new_status = 0
    new_time = ""
    if junction.signal_status == 1:
        new_status = 0
        new_time = "14:00"
    return new_status, new_time

def add_exp_rv(time):
    lambda_ = 1.0
    min_val = 2
    max_val = 10

    random_value = np.random.exponential(scale=1/lambda_)

    scaled_value = int(min_val + (random_value - 0) * (max_val - min_val) / (random_value + 1))  # Shift and scale
    new_time = time_string(time, scaled_value)
    return new_time

def find_halt_time(train, junction):
    if junction.passengerqueue == []:
        return 0
    time_format = "%H:%M"
    halt_time = 0
    arrival_times = []
    for arrival_time, passenger in junction.passengerqueue:
        arrival_times.append(arrival_time)
    rem_capacity = train.capacity - train.no_of_passengers
    if rem_capacity != 0:
        no_of_people_to_board = random.randint(1, int(rem_capacity/100))
    else:
        no_of_people_to_board = 0
        halt_time = 1
        return halt_time

    arrival_times = sorted(arrival_times)
    if no_of_people_to_board < len(arrival_times):
        arrival = arrival_times[no_of_people_to_board - 1]
    else:
        arrival = arrival_times[-1]
    if arrival < train.junc_entering_time:
        halt_time = 1
        return halt_time
    arrival_time = datetime.strptime(arrival, time_format)
    departure_time = datetime.strptime(train.junc_entering_time, time_format)
    time_difference = arrival_time - departure_time
    halt_time = int(time_difference.total_seconds() / 60)
    return halt_time

def adding_new_passengers(train, junction):

    halt_at_station = find_halt_time(train, junction)
    train_departure_time = time_string(train.junc_entering_time, halt_at_station)

    passengers_to_board = []


    for arrival_time, passenger in junction.passengerqueue:
        if train.no_of_passengers >= train.capacity: break
        if arrival_time <= train_departure_time:

            passengers_to_board.append((arrival_time, passenger))
        else:
          continue

    for arrival_time, passenger in passengers_to_board:
        train.passengers.append(passenger)
        train.no_of_passengers += 1
        junction.passengerqueue.remove((arrival_time, passenger))


    return halt_at_station, train, junction

def process_passengers(train, junction, passenger_list):
    for passenger in train.passengers[:]:
        if passenger.destination == junction.name and junction.signal == 1:
            passenger.leaving_time = train.junc_entering_time
            train.no_of_passengers -= 1
            train.passengers.remove(passenger)

    return train, passenger_list

def write_data_to_file(output_file, passenger_data):
    with open(output_file, "w") as file:
        for item in passenger_data:
            file.write(f"{item['name']},{item['line']},{item['arrival_time']},{item['boarding_station']},{item['destination']}\n ")

def updated_file(output_file, passenger_list):
    with open(output_file, 'w') as file:
        for passenger in passenger_list:
            file.write(f"{passenger.__dict__}\n")
            
def find_next_sec_line(line_list, section):
    next_section = "end"

    if section in line_list.sections:
        index = line_list.sections.index(section)
        if index != len(line_list.sections) - 1:
            next_section = line_list.sections[index + 1]
    else:
        next_section = ""
    return next_section

def find_next_junction(current_line, junction):
    next_junction = None    
    for i in range(len(current_line.junctions)-1):
        if current_line.junctions[i].name == junction:
            return current_line.junctions[i+1]
    
    return next_junction

def find_same_junctions(junction, same_junctions):
    for key in same_junctions.keys():
        if (key.signal == junction.signal and key.x == junction.x and key.y == junction.y and key.name!= junction.name):
            return [key]  # Return the list of junctions if found
    return None  

def find_shift_junction(junction, dest_jn, same_junctions):
    curr_line = junction.line 
    for jn in same_junctions:
        if dest_jn.line == jn.line:
            return jn
    return None

def find_shifting_of_passenger(passenger_list, junction_list, line_list):
    for passenger in passenger_list:
        line = find_line_by_number(line_list, passenger.line)
        if not line:
            passenger.shift = 1
            continue
        destination_junction = find_junction_by_name_line(junction_list, passenger.destination, passenger.line)
        if not destination_junction:
            passenger.shift = 1
        else:
            passenger.shift = 0
    
def main():
    
    EPISODES = 100
    action_size=2
    batch_size = 3
    
    train_list = []
    section_list = []
    junction_list = []
    passenger_list = []
    line_list = []
    file_path = "common_junction.txt"

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        print(line.strip()) 
        
    asset_list,same_junctions = obj_creation(file_path,train_list, junction_list, section_list, line_list)
    print(f'train length is {len(train_list)}')

            
    arranging_sections(section_list, line_list)
    arrange_junction(section_list, line_list)
    output_file = "passenger.txt"
    passenger_data=[]
    for junction in junction_list:
        if junction.signal == 1:
            lambda_value = junction.lambda_value if hasattr(junction, 'lambda_value') else 0.5
            num_passengers = junction.num_passengers if hasattr(junction, 'num_passengers') else 10
            passenger_data = generate_passenger_data(passenger_data, junction, lambda_value, num_passengers, line_list,junction_list, output_file)

    write_data_to_file(output_file,passenger_data)
    passenger_list = passenger_parse(output_file)


    for e in range(EPISODES):
        train_list = []
        section_list = []
        junction_list = []
        passenger_list = []
        line_list = []
        file_path = "common_junction.txt"

        with open(file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            print(line.strip()) 
            
        asset_list,same_junctions = obj_creation(file_path,train_list, junction_list, section_list, line_list)
        print(f'train length is {len(train_list)}')

                
        arranging_sections(section_list, line_list)
        arrange_junction(section_list, line_list)

        state_size= get_state(train_list, junction_list).shape[1]  # Returns 24
  
        agent = DQNAgent(state_size, action_size)

        passenger_list = passenger_parse(output_file)

        
        passenger_list, junction_list = passenger_arrival(passenger_list, junction_list)
        find_shifting_of_passenger(passenger_list, junction_list, line_list)
        timetables = []
        for train in train_list:
            print(f"name: {train.name}")
        print(f"train_list length: {len(train_list)}")
        score=0

        while True:
            
            if len(asset_list) == 0: break

            minimum_event_time = asset_list[0].get_event()
            for asset in asset_list:
                if asset.get_event() == "":
                    continue
                if asset.get_event() < minimum_event_time:
                    minimum_event_time = asset.get_event()

            if minimum_event_time >= "14:00": break

            min_asset_list = []
            for asset in asset_list:
                if asset.get_event() == minimum_event_time:
                    min_asset_list.append(asset)

            mod_min_asset_list = {}
            for i in min_asset_list:
                for j in min_asset_list:
                    if i.name == j.name: continue
                    if i.junction == j.junction:
                        print("collision entering the same junction")

            for i in min_asset_list:
                
                state = get_state(train_list,junction_list)
                
                # state = np.reshape(state, [1, state_size])

                action= agent.act(state)
                
                current_junction = find_junction_by_name(junction_list, i.junction)
                current_line = find_line_by_number(line_list, i.line_number)
                spam = False
                if spam:
                    print("hloo")
                else:
                    reward=0
                    halt_time=0
                    if current_junction.signal == 1:
                        new_passengers = []
                        for passenger in i.passengers:
                            if passenger.destination == current_junction.name and passenger.shift==0:
                                passenger.leaving_time = current_junction.train_entering_time
                                boarding_junction = find_junction_by_name_line(junction_list, passenger.boarding_station, passenger.line)
                                destination_junction = find_junction_by_name_line(junction_list, passenger.destination, passenger.line)
                                passenger.delay= passenger.delay + (total_time(passenger.arrival_time, passenger.leaving_time))/get_distance_between_junction (boarding_junction, destination_junction)
                                reward-= passenger.delay
                            elif passenger.shift == 1:
                                boarding_junction = find_junction_by_name_line(junction_list, passenger.boarding_station, passenger.line)
                                destination_junction = None
                                for junc in junction_list:
                                    if junc.name == passenger.destination and junc.signal == 1:
                                        destination_junction = junc
                                        break
                                if not destination_junction:
                                    new_passengers.append(passenger)
                                    continue
                                
                                similar_junctions = find_same_junctions(current_junction, same_junctions)
                                if similar_junctions == None:
                                    new_passengers.append(passenger)
                                else:
                                    shift_junction = find_shift_junction(current_junction, destination_junction, similar_junctions)
                                    if shift_junction == None:
                                        new_passengers.append(passenger)
                                    else:
                                        
                                        passenger.delay = passenger.delay + (total_time(passenger.arrival_time, current_junction.train_leaving_time))/get_distance_between_junction(boarding_junction, current_junction)
                                        reward-= passenger.delay
                                        passenger.arrival_time = current_junction.train_leaving_time
                                        shift_junction.passengerqueue.append((passenger.arrival_time, passenger))
                                        passenger.line = shift_junction.line
                                        passenger.shift = 0
                                        passenger.boarding_station = shift_junction.name 
                                        
                            else: new_passengers.append(passenger)

                        i.passengers = new_passengers
                        i.no_of_passengers = len(i.passengers)

                        halt_time, i, current_junction = adding_new_passengers(i, current_junction)
                        i.update_passenger_at_time(i.junc_leaving_time)

                    score+=reward
                    if action==1:
                        next_state= state
                        halt_time= 3
                        i.junc_leaving_time = time_string(i.junc_leaving_time, int(halt_time))
                        agent.memorize(state, action, reward, next_state)

                        continue 
                    
                    current_junction.train = ""
                    current_junction.train_entering_time = ""
                    current_junction.train_leaving_time = ""
                    

                    i.junc_leaving_time = time_string(i.junc_leaving_time, int(halt_time))
                    if isinstance(i, Trains):
                        for train in asset_list:
                            print(f"Name: {train.name}, current_junction: {train.junction}, entering_time: {train.junc_entering_time}, leaving_time: {train.junc_leaving_time}")

                    if find_next_junction(current_line, i.junction)  == None:
                        asset_list.remove(i)
                        continue
                    new_junction= find_next_junction(current_line, current_junction.name)

                    section_len = distance(new_junction.x, new_junction.y, current_junction.x, current_junction.y)
                    i.junction = new_junction.name

                    new_junc_entering_time = time_string(i.junc_leaving_time, int((60 * section_len/i.speed)))
                    new_junc_leaving_time = new_junc_entering_time
                    i.junc_entering_time = new_junc_entering_time
                    i.junc_leaving_time = new_junc_leaving_time
                    new_junction.train = i.name
                    new_junction.train_entering_time = new_junc_entering_time
                    new_junction.train_leaving_time = new_junc_leaving_time
                    
                    next_state = get_state(train_list, junction_list)
                    agent.memorize(state, action, reward, next_state)
                    timetables.append(timetable(i.name, i.junction, i.junc_entering_time, i.junc_leaving_time))



        final_passenger= "final_passenger.txt"
        updated_file(final_passenger,passenger_list)
        if len(agent.memory) > batch_size:
            print(f'Replaying')
            agent.replay(batch_size)
        # agent.model.save(f"dqn_model_episode_{e}.h5")

        print("episode: {}/{}, score: {}".format(e, EPISODES, score))
        doc_file = "Rewards.txt"
        with open(doc_file, "a") as f:
            f.write(f"episode: {e}/{EPISODES}, score: {score}\n")

    for train in train_list:
        time_series = pd.to_datetime([entry[0] for entry in train.passenger_time], format="%H:%M")
        passenger_counts = [entry[1] for entry in train.passenger_time]
            
        plt.plot(time_series, passenger_counts, label=train.name)  # Train name will appear in the legend

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  

    plt.xlabel('Time')
    plt.ylabel('Number of Passengers')
    plt.title('Passengers in Train Over Time')

    plt.legend()
    plt.savefig("plot.png")




if __name__ == "__main__":
    main()

