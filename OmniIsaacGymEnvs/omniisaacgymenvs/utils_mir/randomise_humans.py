import random



def rand_spawn(spawns, n):
    num_objects = n

    # Initialize lists to store spawn and destination positions

    spawnkey = list(spawns.values())
    random.shuffle(spawnkey)

    spawn_positions  = spawnkey[:num_objects]

    return spawn_positions

def assign_destinations(spawns, destinations):
    assigned_destinations = {}  # Dictionary to store assigned destinations
    
    for i, spawn in enumerate(spawns):
        x, y, spawn_type = spawn

        # Check if the spawn type is 'in' or 'out'
        if spawn_type == 'in':
            # Assign an 'out' destination
            assigned_destinations[i] = random.choice([value for value in destinations.values() if (value[2] == 'out' or (value[0] == x and value[1] == y))])
            
        else:
            # Assign an 'in' destination
            assigned_destinations[i] = random.choice([value for value in destinations.values() if value[2] == 'in' or (value[0] == x and value[1] == y)])
        for i, dest in assigned_destinations.items():
            if dest in destinations.values():
                key_to_remove = next(key for key, value in destinations.items() if value == dest)
                del destinations[key_to_remove]

    return assigned_destinations

# Define the spawn and destination coordinates
def set_humans(n):
    spawns = {
        0: [-20.0, 5.0, 'out'],
        1: [ 0.5, 7.5, 'in'],
        2: [-0.5, 8.5, 'in'],
        3: [-0.5, 7.5, 'in'],
        4: [-0.5, 9.5, 'in'],
        5: [0.5, 8.5, 'in'],
        6: [0.5, 9.5, 'in'],
        7: [-14.5, 5.0, 'out']
    }

    destinations = {
        0: [20.0, 5.5, 'out'],
        1: [ 0.5, 7.5, 'in'],
        2: [-0.5, 8.5, 'in'],
        3: [-0.5, 7.5, 'in'],
        4: [-0.5, 9.5, 'in'],
        5: [0.5, 8.5, 'in'],
        6: [0.5, 9.5, 'in'],
        7: [-24.5, 4.5, 'out'],
        8: [-24.5, 3.5, 'out'],
        9: [-23.5, 3.5, 'out']
    }


    spawn_positions = rand_spawn(spawns, n)
    assigned_destinations = assign_destinations(spawn_positions, destinations)
    spawn_dict = {}
    dest_dict = {}
    for i in range(len(spawn_positions)):
        spawn_dict[i] = [spawn_positions[i][0],spawn_positions[i][1]]
        dest_dict[i] = [assigned_destinations[i][0], assigned_destinations[i][1]]
    
    return spawn_dict,dest_dict, assigned_destinations




def get_human_cmd(n):
    spawn_dict , dest_dict, assigned_destinations = set_humans(n)
    command = ''
# Print the assigned destinations
    for i, dest in assigned_destinations.items():
    #    print(f"Spawn {spawn_positions[i]}: Destination {dest}")
        print(f"Spawn Human_n{i} {spawn_dict[i]}")
        command += f"Spawn Human_n{i} {spawn_dict[i][0]} {spawn_dict[i][1]} 0 0 \n"

    for i, dest in assigned_destinations.items():
    #    print(f"Spawn {spawn_positions[i]}: Destination {dest}")
        print(f"Human_n{i} Idle {random.randint(1,8)}")
        command += f"Human_n{i} Idle {random.randint(0,8)}\n"


    for i, dest in assigned_destinations.items():
    #    print(f"Spawn {spawn_positions[i]}: Destination {dest}")
        print(f"Human_n{i} GoTo {dest_dict[i]}")
        command += f"Human_n{i} GoTo {dest_dict[i][0]} {dest_dict[i][1]} 0 _ \n"

    return command

print(get_human_cmd(5))