import numpy as np
from environment import MonkeyBananaEnv


# After an agent has been trained, env_helper should
# make a couple of assertions that calculate various diagnostics such as
# total number of unique states, etc.
# and also percentage of correct state transitions in the model, and return all of that
def check_banana_on_floor(env, agent):
    unique_states = set()
    for key, value in agent.model.items():
        unique_states.add(key[0])
        unique_states.add(value[1])

    all_states = generate_banana_on_floor_states(env.size)
    missing_states = all_states - unique_states

    filtered_model = agent.model
    # Check correctness
    for key, value in agent.model.items():
        # Go right or go left actions
        if key[1] in [0, 1]:
            filtered_model[key] = value

    num_correct = 0
    for key, value in filtered_model.items():
        initial_state = env.index_to_state(key[0])
        action = env.action_index_to_label(key[1])
        next_state = env.index_to_state(value[1])

        correct = (
            (
                np.clip(
                    initial_state["agent"][0] + (1 if action == "right" else -1), 1, 5
                )
                == next_state["agent"][0]
            )
            and initial_state["agent"][1] == next_state["agent"][1]
            and initial_state["chair"][0] == next_state["chair"][0]
            and initial_state["chair"][1] == next_state["chair"][1]
            and initial_state["banana"][0] == next_state["banana"][0]
            and initial_state["banana"][1] == next_state["banana"][1]
        )
        if correct:
            num_correct += 1

    correctness = num_correct / len(filtered_model)

    return {
        "Total states": len(all_states),
        "Unique states in model": len(unique_states),
        "Missing states": len(missing_states),
        "Model accuracy": correctness,
    }


def generate_banana_on_floor_states(size=5):
    # 5 \* 5 \* 5 (when monkey is at height 1) +
    # 5 \* 5 (when monkey is at height 2) +
    # = 150

    states = set()
    # Monkey on floor
    for monkey_x in range(1, size + 1):
        for chair_x in range(1, size + 1):
            for banana_x in range(1, size + 1):
                # Concatenate the state into a string
                states.add(
                    int(
                        str(monkey_x)
                        + str(1)
                        + str(chair_x)
                        + str(1)
                        + str(banana_x)
                        + str(2)
                    )
                )

    # # Monkey on chair
    # for monkey_x in range(1, size + 1):
    #     for banana_x in range(1, size + 1):
    #         states.add(
    #             int(
    #                 str(monkey_x)
    #                 + str(2)
    #                 + str(monkey_x)
    #                 + str(1)
    #                 + str(banana_x)
    #                 + str(2)
    #             )
    #         )

    return states


def check_reach_banana_with_chair(env, agent):
    unique_states = set()
    for key, value in agent.model.items():
        unique_states.add(key[0])
        unique_states.add(value[1])

    # Number of states is same as the first environment
    all_states = generate_banana_on_floor_states(env.size)
    missing_states = all_states - unique_states

    filtered_model = {}
    # Check correctness
    for key, value in agent.model.items():
        # Go right or go left actions
        # Go right with chair or go left with chair actions
        if key[1] in [0, 1, 2, 3]:
            filtered_model[key] = value

    num_correct = 0
    errors = []
    for key, value in filtered_model.items():
        initial_state = env.index_to_state(key[0])
        action = env.action_index_to_label(key[1])
        next_state = env.index_to_state(value[1])

        predicted_agent_x = np.clip(
            initial_state["agent"][0]
            + (
                1
                if (
                    action == "right"
                    or (
                        action == "right with chair"
                        and initial_state["agent"][0] == initial_state["chair"][0]
                    )
                )
                else -1
                if (
                    action == "left"
                    or (
                        action == "left with chair"
                        and initial_state["agent"][0] == initial_state["chair"][0]
                    )
                )
                else 0
            ),
            1,
            5,
        )
        predicted_chair_x = np.clip(
            initial_state["chair"][0]
            + (
                1
                if (
                    action == "right with chair"
                    and initial_state["agent"][0] == initial_state["chair"][0]
                )
                else -1
                if (
                    action == "left with chair"
                    and initial_state["agent"][0] == initial_state["chair"][0]
                )
                else 0
            ),
            1,
            5,
        )

        correct = (
            predicted_agent_x == next_state["agent"][0]
            and initial_state["agent"][1] == next_state["agent"][1]
            and predicted_chair_x == next_state["chair"][0]
            and initial_state["chair"][1] == next_state["chair"][1]
            and initial_state["banana"][0] == next_state["banana"][0]
            and initial_state["banana"][1] == next_state["banana"][1]
        )

        if correct:
            num_correct += 1
        else:
            errors.append(
                {
                    "initial_state": initial_state,
                    "action": action,
                    "next_state": next_state,
                    "predicted_agent_x": predicted_agent_x,
                    "predicted_chair_x": predicted_chair_x,
                }
            )

    correctness = num_correct / len(filtered_model)

    return {
        "Total states": len(all_states),
        "Unique states in model": len(unique_states),
        "Missing states": len(missing_states),
        "Model accuracy": correctness,
        "Errors": errors,
    }


def check_climb_to_reach_banana(env, agent):
    unique_states = set()
    for key, value in agent.model.items():
        unique_states.add(key[0])
        unique_states.add(value[1])

    # Number of states is same as the first environment
    all_states = generate_climb_to_reach_banana_states(env.size)
    missing_states = all_states - unique_states

    filtered_model = {}
    # Check correctness
    for key, value in agent.model.items():
        # Go right or go left actions
        # Go right with chair or go left with chair actions
        if key[1] == 4:
            filtered_model[key] = value

    num_correct = 0
    errors = []
    for key, value in filtered_model.items():
        initial_state = env.index_to_state(key[0])
        action = env.action_index_to_label(key[1])
        next_state = env.index_to_state(value[1])

        predicted_agent_y = np.clip(initial_state["agent"][1] + 1, 1, 2)

        correct = (
            initial_state["agent"][0] == next_state["agent"][0]
            and predicted_agent_y == next_state["agent"][1]
            and initial_state["chair"][0] == next_state["chair"][0]
            and initial_state["chair"][1] == next_state["chair"][1]
            and initial_state["banana"][0] == next_state["banana"][0]
            and initial_state["banana"][1] == next_state["banana"][1]
        )

        if correct:
            num_correct += 1
        else:
            errors.append(
                {
                    "initial_state": initial_state,
                    "action": action,
                    "next_state": next_state,
                    "predicted_agent_y": predicted_agent_y,
                }
            )

    correctness = num_correct / len(filtered_model)

    return {
        "Total states": len(all_states),
        "Unique states in model": len(unique_states),
        "Missing states": len(missing_states),
        "Model accuracy": correctness,
        "Errors": errors,
    }


def generate_climb_to_reach_banana_states(size=5):
    states = set()
    for monkey_x in range(1, size + 1):
        states.add(
            int(
                str(monkey_x) + str(1) + str(monkey_x) + str(1) + str(monkey_x) + str(2)
            )
        )

    for monkey_x in range(1, size + 1):
        states.add(
            int(
                str(monkey_x) + str(2) + str(monkey_x) + str(1) + str(monkey_x) + str(2)
            )
        )

    return states


def check_climb_down(env, agent):
    unique_states = set()
    for key, value in agent.model.items():
        unique_states.add(key[0])
        unique_states.add(value[1])

    # Number of states is same as the climb environment
    all_states = generate_climb_to_reach_banana_states(env.size)
    missing_states = all_states - unique_states

    filtered_model = {}
    # Check correctness
    for key, value in agent.model.items():
        if key[1] == 5:
            filtered_model[key] = value

    num_correct = 0
    errors = []
    for key, value in filtered_model.items():
        initial_state = env.index_to_state(key[0])
        action = env.action_index_to_label(key[1])
        next_state = env.index_to_state(value[1])

        predicted_agent_y = np.clip(initial_state["agent"][1] - 1, 1, 2)

        correct = (
            initial_state["agent"][0] == next_state["agent"][0]
            and predicted_agent_y == next_state["agent"][1]
            and initial_state["chair"][0] == next_state["chair"][0]
            and initial_state["chair"][1] == next_state["chair"][1]
            and initial_state["banana"][0] == next_state["banana"][0]
            and initial_state["banana"][1] == next_state["banana"][1]
        )

        if correct:
            num_correct += 1
        else:
            errors.append(
                {
                    "initial_state": initial_state,
                    "action": action,
                    "next_state": next_state,
                    "predicted_agent_y": predicted_agent_y,
                }
            )

    correctness = num_correct / len(filtered_model)

    return {
        "Total states": len(all_states),
        "Unique states in model": len(unique_states),
        "Missing states": len(missing_states),
        "Model accuracy": correctness,
        "Errors": errors,
    }


def check_full_model(env, agent):
    unique_states = set()
    for key, value in agent.model.items():
        unique_states.add(key[0])
        unique_states.add(value[1])

    all_states = generate_monkey_banana_states(env.size)
    missing_states = all_states - unique_states

    num_correct = 0
    errors = []
    for key, value in agent.model.items():
        initial_state = env.index_to_state(key[0])
        action = env.action_index_to_label(key[1])
        next_state = env.index_to_state(value[1])

        env = MonkeyBananaEnv(size=5)
        env.reset(start_state=initial_state)
        next_state_from_env, reward, terminated, truncated, _ = env.step(key[1])

        correct = env.state_to_index(next_state) == next_state_from_env

        if correct:
            num_correct += 1
        else:
            errors.append(
                {
                    "initial_state": initial_state,
                    "action": action,
                    "next_state": next_state,
                }
            )

    correctness = num_correct / len(agent.model)

    return {
        "Total states": len(all_states),
        "Unique states in model": len(unique_states),
        "Missing states": missing_states,
        "Model accuracy": correctness,
        "Errors": errors,
    }


# All the possible final states
def generate_monkey_banana_states(size=5):
    # Monkey is on ground: 5 * 5 * 5 = 125
    # Monkey is on chair: 5 * 5 = 25

    states = set()
    # Monkey on floor
    for monkey_x in range(1, size + 1):
        for chair_x in range(1, size + 1):
            for banana_x in range(1, size + 1):
                # Concatenate the state into a string
                states.add(
                    int(
                        str(monkey_x)
                        + str(1)
                        + str(chair_x)
                        + str(1)
                        + str(banana_x)
                        + str(2)
                    )
                )

    # Monkey on chair
    for monkey_x in range(1, size + 1):
        for banana_x in range(1, size + 1):
            states.add(
                int(
                    str(monkey_x)
                    + str(2)
                    + str(monkey_x)
                    + str(1)
                    + str(banana_x)
                    + str(2)
                )
            )

    return states
