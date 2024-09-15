module TDLearning

using Random
using CSV
using JSON
using DataFrames

mutable struct TDLearner
    Q1::Dict{Tuple{Tuple{Int, Int}, Int}, Float64}  # Q-values table for Snake 1
    Q2::Dict{Tuple{Tuple{Int, Int}, Int}, Float64}  # Q-values table for Snake 2
    α::Float64  # Learning rate (0-1)
    γ::Float64  # Discount factor (0-1)
    ε1::Float64  # Exploration rate for Snake 1 (0-1)
    ε2::Float64  # Exploration rate for Snake 2 (0-1)
end

function TDLearner(; α = 0.01, γ = 0.997, ε1 = 0.1, ε2 = 0.1)
    Q1 = Dict{Tuple{Tuple{Int, Int}, Int}, Float64}()
    Q2 = Dict{Tuple{Tuple{Int, Int}, Int}, Float64}()
    return TDLearner(Q1, Q2, α, γ, ε1, ε2)
end

const NUM_ACTIONS = 4

# Modify the choose_action function to use separate Q-tables for each snake
function choose_action(env, learner::TDLearner, state, agent::Int)
    if agent == 1
        if rand() < learner.ε1
            return rand(1:NUM_ACTIONS)  # Explore for Snake 1
        else
            q_values = [get(learner.Q1, (state, a), 0.0) for a in 1:NUM_ACTIONS]
            return argmax(q_values)  # Exploit for Snake 1
        end
    elseif agent == 2
        if rand() < learner.ε2
            return rand(1:NUM_ACTIONS)  # Explore for Snake 2
        else
            q_values = [get(learner.Q2, (state, a), 0.0) for a in 1:NUM_ACTIONS]
            return argmax(q_values)  # Exploit for Snake 2
        end
    end
end

# Separate update_Q function for each snake
function update_Q(learner::TDLearner, state::Tuple{Int, Int}, action::Int, reward, next_state::Tuple{Int, Int}, agent::Int)
    next_state_key = (next_state[1], next_state[2])
    state_key = (state[1], state[2])
    
    if agent == 1
        best_next_action = argmax([get(learner.Q1, ((next_state_key[1], next_state_key[2]), a), 0.0) for a in 1:NUM_ACTIONS])
        td_target = reward + learner.γ * get(learner.Q1, ((next_state_key[1], next_state_key[2]), best_next_action), 0.0)
        learner.Q1[((state_key[1], state_key[2]), action)] = get(learner.Q1, ((state_key[1], state_key[2]), action), 0.0) + learner.α * (td_target - get(learner.Q1, ((state_key[1], state_key[2]), action), 0.0))
    elseif agent == 2
        best_next_action = argmax([get(learner.Q2, ((next_state_key[1], next_state_key[2]), a), 0.0) for a in 1:NUM_ACTIONS])
        td_target = reward + learner.γ * get(learner.Q2, ((next_state_key[1], next_state_key[2]), best_next_action), 0.0)
        learner.Q2[((state_key[1], state_key[2]), action)] = get(learner.Q2, ((state_key[1], state_key[2]), action), 0.0) + learner.α * (td_target - get(learner.Q2, ((state_key[1], state_key[2]), action), 0.0))
    end
end

# Modify run_td_learning! to use separate Q-values and exploration rates for each snake
function run_td_learning!(env, learner::TDLearner, num_episodes::Int, csv_filename::String)
    rewards = DataFrame(Episode = 1:num_episodes, Reward1 = zeros(Float64, num_episodes), Reward2 = zeros(Float64, num_episodes))
    survival_times = DataFrame(Episode = 1:num_episodes, SurvivalTime1 = zeros(Int, num_episodes), SurvivalTime2 = zeros(Int, num_episodes))
    food_consumed = DataFrame(Episode = 1:num_episodes, Food1 = zeros(Int, num_episodes), Food2 = zeros(Int, num_episodes))

    position_counts1 = zeros(Int, size(env.tile_map, 2), size(env.tile_map, 3))  # For Snake 1
    position_counts2 = zeros(Int, size(env.tile_map, 2), size(env.tile_map, 3))  # For Snake 2

    best_survival_time = 0
    best_game_states = []

    for episode in 1:num_episodes
        Main.GridWorlds.reset!(env)
        state1 = (env.agent1_position[1], env.agent1_position[2])
        state2 = (env.agent2_position[1], env.agent2_position[2])

        episode_reward1 = 0.0
        episode_reward2 = 0.0
        steps1 = 0
        steps2 = 0
        food1 = 0
        food2 = 0
        reward1 = 0
        reward2 = 0

        game_states = []

        while !env.done
            game_state = Dict(
                "agent1_position" => (env.agent1_position[1], env.agent1_position[2]),
                "agent2_position" => (env.agent2_position[1], env.agent2_position[2]),
                "food_position" => (env.food_position[1], env.food_position[2]),
                "tile_map" => copy(env.tile_map)
            )
            push!(game_states, game_state)

            if env.alive1

                position_counts1[env.agent1_position[1], env.agent1_position[2]] += 1

                action1 = choose_action(env, learner, state1, 1)  # Pass state as tuple
                Main.GridWorlds.act!(env, action1, 1)
                next_state1 = (env.agent1_position[1], env.agent1_position[2])  # Convert CartesianIndex to tuple
                if env.reward1 - reward1 >= 2  # Assuming positive reward indicates food consumption
                    food1 += 1
                 
                end
                reward1 = env.reward1
                update_Q(learner, state1, action1, reward1, next_state1, 1)
                state1 = next_state1
                episode_reward1 += reward1
                steps1 += 1

            end

            if env.alive2

                position_counts2[env.agent2_position[1], env.agent2_position[2]] += 1

                action2 = choose_action(env, learner, state2, 2)  # Pass state as tuple
                Main.GridWorlds.act!(env, action2, 2)
                next_state2 = (env.agent2_position[1], env.agent2_position[2])  # Convert CartesianIndex to tuple
                if env.reward2 - reward2 >= 2  # Assuming positive reward indicates food consumption
                    food2 += 1

                end
                reward2 = env.reward2
                update_Q(learner, state2, action2, reward2, next_state2, 2)
                state2 = next_state2
                episode_reward2 += reward2
                steps2 += 1

            end

            if steps1 > 200 || steps2 > 200 || episode_reward1 < -1000 || episode_reward2 < -1000
                env.done = true
            end

            # Ensure rewards are printed correctly for debugging
            println("Reward1: $(env.reward1), Reward2: $(env.reward2)")
        end

        # Decay the exploration rate
        learner.ε1 *= 0.99  # Adjust the decay rate for Snake 1
        learner.ε2 *= 0.99  # Adjust the decay rate for Snake 2

        rewards[episode, :] .= (episode, episode_reward1, episode_reward2)
        survival_times[episode, :] .= (episode, steps1, steps2)
        food_consumed[episode, :] .= (episode, env.food1, env.food2)

        if steps1 + steps2 > best_survival_time
            best_survival_time = steps1 + steps2
            best_game_states = game_states
        end

        println("Episode $episode finished.")
    end

    df_position_counts1 = DataFrame(position_counts1, :auto)
    df_position_counts2 = DataFrame(position_counts2, :auto)

    # Save the DataFrames to CSV files
    CSV.write("position_counts1_$csv_filename.csv", df_position_counts1)
    CSV.write("position_counts2_$csv_filename.csv", df_position_counts2)

    CSV.write(csv_filename, rewards)
    CSV.write("survival_times_$csv_filename", survival_times)
    CSV.write("food_consumed_$csv_filename", food_consumed)

    open("best_game_states.json", "w") do f
        JSON.print(f, best_game_states)
    end

end


end # module
