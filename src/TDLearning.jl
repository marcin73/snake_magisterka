module TDLearning

using Random

struct TDLearner
    Q::Dict{Tuple{Tuple{Int, Int}, Int}, Float64}  # Q-values table
    α::Float64  # Learning rate (0-1)
    γ::Float64  # Discount factor (0-1)
    ε::Float64  # Exploration rate (0-1)
end

function TDLearner(; α = 0.15, γ = 0.95, ε = 0.03)
    Q = Dict{Tuple{Tuple{Int, Int}, Int}, Float64}()
    return TDLearner(Q, α, γ, ε)
end

const NUM_ACTIONS = 4

function choose_action(env, learner::TDLearner, state, agent)
    if rand() < learner.ε
        return rand(1:NUM_ACTIONS)  # Explore, 1 : 4 num_actions hardcoded bo cos sie nie pobeira
    else
        q_values = [get(learner.Q, (state, a), 0.0) for a in 1:NUM_ACTIONS]
        return argmax(q_values)  # Exploit jw 
    end
end

function update_Q(learner::TDLearner, state::Tuple{Int, Int}, action::Int, reward, next_state::Tuple{Int, Int})
    next_state_key = (next_state[1], next_state[2])
    state_key = (state[1], state[2])
    
    best_next_action = argmax([get(learner.Q, ((next_state_key[1], next_state_key[2]), a), 0.0) for a in 1:NUM_ACTIONS])
    td_target = reward + learner.γ * get(learner.Q, ((next_state_key[1], next_state_key[2]), best_next_action), 0.0)
    learner.Q[((state_key[1], state_key[2]), action)] = get(learner.Q, ((state_key[1], state_key[2]), action), 0.0) + learner.α * (td_target - get(learner.Q, ((state_key[1], state_key[2]), action), 0.0))
end




using CSV
using DataFrames

function run_td_learning!(env, learner::TDLearner, num_episodes::Int, csv_filename::String)
    rewards = DataFrame(Episode = 1:num_episodes, Reward1 = zeros(Float64, num_episodes), Reward2 = zeros(Float64, num_episodes))

    for episode in 1:num_episodes
        Main.GridWorlds.reset!(env)
        state1 = (env.agent1_position[1], env.agent1_position[2])
        state2 = (env.agent2_position[1], env.agent2_position[2])

        #poki co bezuzyteczne
        episode_reward1 = 0.0
        episode_reward2 = 0.0

        while !env.done
            if env.alive1 #tylko jak zyje
                action1 = choose_action(env, learner, state1, 1)  # akcja jako tupla
                Main.GridWorlds.act!(env, action1, 1)
                next_state1 = (env.agent1_position[1], env.agent1_position[2])  # zmina na tuple
                reward1 = env.reward1
                update_Q(learner, state1, action1, reward1, next_state1)
                state1 = next_state1
                episode_reward1 += reward1
            end

            if env.alive2 #tylko jak zyje
                action2 = choose_action(env, learner, state2, 2)  # jw
                Main.GridWorlds.act!(env, action2, 2)
                next_state2 = (env.agent2_position[1], env.agent2_position[2])  # jw
                reward2 = env.reward2
                update_Q(learner, state2, action2, reward2, next_state2)
                state2 = next_state2
                episode_reward2 += reward2
            end

            # debbug. print rewarda a nie episode_reward, bo tamten jest kumulowany i wychodza glupoty.
            println("Reward1: $(env.reward1), Reward2: $(env.reward2)")
            println("State1: $state1, State2: $state2")
            println("Nagroda: $(env.food_position)")
        end

      
        #pod csv dataframe
        rewards[episode, :] .= (episode, episode_reward1, episode_reward2)
        println("Episode $episode finished.")
    end

    CSV.write(csv_filename, rewards)
end




end # module
