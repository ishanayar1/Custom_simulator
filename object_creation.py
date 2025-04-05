class Trains:
    def __init__(self, name, line_number, start_time, speed, capacity):
        self.line_number = line_number
        self.name = name
        self.start_time = start_time
        self.speed = speed
        self.capacity = capacity
        self.no_of_passengers = 0
        self.passengers = []
        self.section = ""
        self.sec_leaving_time = ""
        self.sec_entering_time = ""
        self.passenger_time = []

    def get_event(self):
        return self.sec_leaving_time

    def set_event(self, new_section, new_sec_entering_time, new_sec_leaving_time):
        self.section = new_section
        self.sec_entering_time = new_sec_entering_time
        self.sec_leaving_time = new_sec_leaving_time

    def set_passengers(self, new_passengers):
        self.no_of_passengers = new_passengers

    def update_passenger_at_time(self, time):
        self.passenger_time.append((time, self.no_of_passengers))

class Section:
    def __init__(self, name, line_number, junction_start, junction_end):

        self.line_number = line_number
        self.name = name
        self.start = junction_start
        self.end = junction_end
        self.next_section = ""
        self.train = ""
        self.train_leaving_time = ""
        self.train_entering_time = ""

    def get_event(self):
        return self.train_leaving_time

    def set_event(self, new_train, new_train_starting_time, new_train_leaving_time):
        self.train = new_train
        self.train_starting_time = new_train_starting_time
        self.train_leaving_time = new_train_leaving_time

class Junction:
    def __init__(self, name, line, signal, x, y,lambda_value=0.5, num_passengers=10):
        self.name = name
        self.line = line
        self.signal = signal
        self.x = x
        self.y = y
        self.signal_status = 0
        self.lambda_value = lambda_value
        # self.rev_lambda_value = lambda_value
        self.num_passengers = num_passengers
        self.time_of_change = "12:00"
        self.passengerqueue = []

    def get_status(self):
        return self.signal_status

    def get_event(self):
        return self.time_of_change

    def set_status(self, new_signal, new_time_of_change):
        self.signal_status = new_signal
        self.time_of_change = new_time_of_change

class Line:
    def __init__(self, line_number):
        self.line_number = line_number
        self.sections = []
        self.junctions = []

    def add_section(self, section):
        self.sections.append(section)

    def add_junction(self, junction):
        self.junctions.append(junction)

def find_junction_by_name_line(junction_list, target_name, target_line):
    for junction in junction_list:
        if junction.name == target_name and abs(junction.line) == abs(target_line):
            return junction
    return None

def find_train_by_name(train_list, target_name):
    for train in train_list:
        if train.name == target_name:
            return train
    return None

def find_section_by_name(section_list, target_name):
    for section in section_list:
        if section.name == target_name:
            return section
    return None

def find_line_by_number(line_list, line_number):
    for line in line_list:
        if line.line_number == line_number:
            return line
    return None   

def obj_creation(file_path,train_list, junction_list, section_list, line_list):
    dummy_secs_for_lines = {}
    with open(file_path, "r") as file:
        lines = file.readlines()


    mode = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower() == "trains":
            mode = "trains"
            continue
        elif line.lower() == "junctions":
            mode = "junctions"
            continue
        elif line.lower() == "sections":
            mode = "sections"
            continue
        elif line.lower() == "lines and first sections":
            mode = "first sections"
            continue

        if mode == "trains":
            name, line_number, start_time, speed, capacity = line.split(",")
            train = Trains(name, int(line_number), start_time, float(speed), int(capacity))
            train_list.append(train)

        elif mode == "junctions":
          parts = line.split(",")
          name, line_number, signal, x, y = parts[:5]
          lambda_value = float(parts[5]) if len(parts) > 5 else 0
          no_of_passenger = float(parts[6]) if len(parts) > 6 else 0

          junction = Junction(name, int(line_number), int(signal), float(x), float(y), lambda_value, no_of_passenger)
          junction_list.append(junction)

        elif mode == "sections":
            name, line_number, junction_start, junction_end = [x.strip() for x in line.split(",")]
            start_junc = find_junction_by_name_line(junction_list, junction_start, int(line_number))
            if start_junc == None:
                print("does not exist start junction")
            end_junc = find_junction_by_name_line(junction_list, junction_end, int(line_number))
            if end_junc == None:
                print("does not exist end junction")
            section = Section(name, int(line_number), start_junc, end_junc)
            section_list.append(section)

        elif mode == "first sections":
            line, first_section = line.split(",")
            line = int(line)

            line_obj = find_line_by_number(line_list, line)
            first_section_obj = find_section_by_name(section_list, first_section)

            if first_section_obj == None:
                print("section does not exist")
            else:
                print(f"first_section: {first_section_obj.name}, line:{line}")

            if line_obj == None:
                new_line_obj = Line(line)
                dummy_section_obj = Section(f"dummy_{line}", line, first_section_obj.start, first_section_obj.start)
                dummy_section_obj.next_section = first_section_obj.name
                new_line_obj.sections.append(dummy_section_obj.name)
                line_list.append(new_line_obj)
                section_list.append(dummy_section_obj)

    # for sec in section_list:
    #     print(f"name: {sec.name}")

    for train in train_list:
        train_line = find_line_by_number(line_list, train.line_number)
        if train_line == None:
            print("line does not exist")
        first_section = find_section_by_name(section_list, train_line.sections[0])
        if first_section == None:
            print("first section does not exist")
        train.section = first_section.name
        train.sec_entering_time = train.start_time
        train.sec_leaving_time = train.start_time

        if first_section.train_entering_time == "" or first_section.train_entering_time > train.start_time:
          first_section.train = train.name
          first_section.train_leaving_time = train.start_time
          first_section.train_entering_time = train.start_time

    # for j in junction_list:
    #     print(f"Junction: {j.name}, Coordinates: ({j.x}, {j.y})")


    signal_list = []
    for j in junction_list:
        if j.signal == 1:
            signal_list.append(j)
    for train in train_list:
          print(f"Name: {train.name}, current_section: {train.section}, entering_time: {train.sec_entering_time}, leaving_time: {train.sec_leaving_time}")

    common_junctions= find_common_junction(junction_list)
    assets = train_list + signal_list
    return assets, common_junctions


def find_common_junction(junction_list):
    junction_map = {}
    for i in junction_list:
        for j in junction_list:
            if i.name == j.name:
                continue
            elif i.line == j.line:
                continue
            elif i.x == j.x and i.y == j.y:
                if i not in junction_map:
                    junction_map[i] = set()  
                if j not in junction_map:
                    junction_map[j] = set()
                
                junction_map[i].add(j)
                junction_map[j].add(i)  

    printed_junctions = set()
    for i, j in junction_map.items():
        if i in printed_junctions:
            continue  
        print(f"{i.name} (Line {i.line}) and common are:")
        for common in j:
            print(f" {common.name} (Line {common.line})")
            printed_junctions.add(common)  
        printed_junctions.add(i)  
    return junction_map

def arranging_sections(section_list, line_list):
    line_sec_dict = {}
    for i in section_list:
        if i.line_number <0: continue
        if i.line_number not in line_sec_dict:
            line_sec_dict[i.line_number] = []
        line_sec_dict[i.line_number].append(i)

    for key in line_sec_dict:
        last_section = 0
        for i in line_sec_dict[key]:
            one_and_only = 0
            for j in line_sec_dict[key]:
                if i.end.x == j.start.x and i.end.y == j.start.y:
                    if one_and_only == 1: print("more than one section")
                    else:
                        one_and_only = 1
                        i.next_section = j.name
                    if i.end is None:
                        print(f"ERROR: i.end is None for section {i.start}")
                    if j.start is None:
                        print(f"ERROR: j.start is None for section {j.end}")

            if one_and_only == 0:
                if last_section == 1: print("more than one end section")
                else:
                    last_section = 1
                    i.next_section = "end"

    for line in line_sec_dict:
        line_obj = find_line_by_number(line_list, line)
        if line_obj == None:
          print("line does not exist")
          continue
        current_section = line_obj.sections[0]
        current_junction = ""
        while current_section != "end":
            current_section_obj = find_section_by_name(section_list, current_section)
            current_junction = current_section_obj.end
            line_obj.junctions.append(current_junction)
            current_section = current_section_obj.next_section
            line_obj.sections.append(current_section)



    return


# encode state space in vector, junction , histogram of passenger waiting. example junction:2 cross section.
# put stochastic delay. how many people switching train

#error: passenger boarding and destination