module SnakeModule

import DataStructures as DS
import ..GridWorlds as GW
import Random
import ReinforcementLearningBase as RLBase

#####
##### game logic
#####

const NUM_OBJECTS = 6  # Correct number of layers
const AGENT1 = 1
const WALL = 2
const BODY1 = 3
const FOOD = 4
const AGENT2 = 5  # Second agent
const BODY2 = 6  # Second agent's body
const NUM_ACTIONS = 4

mutable struct Snake{R, RNG} <: GW.AbstractGridWorld
    tile_map::BitArray{3}
    agent1_position::CartesianIndex{2}
    agent2_position::CartesianIndex{2}
    reward1::R
    reward2::R
    rng::RNG
    done::Bool
    terminal_reward::R
    terminal_penalty::R
    food_reward::R
    food_position::CartesianIndex{2}
    body1::DS.Queue{CartesianIndex{2}}
    body2::DS.Queue{CartesianIndex{2}}
    alive1::Bool # czy zyje waz 1
    alive2::Bool # czy zyuje waz 2
end

function Snake(; R = Float32, height = 8, width = 8, rng = Random.GLOBAL_RNG)
    tile_map = falses(NUM_OBJECTS, height, width)

    tile_map[WALL, 1, :] .= true
    tile_map[WALL, height, :] .= true
    tile_map[WALL, :, 1] .= true
    tile_map[WALL, :, width] .= true

    agent1_position = GW.sample_empty_position(rng, tile_map)
    tile_map[AGENT1, agent1_position] = true
    body1 = DS.Queue{CartesianIndex{2}}()
    DS.enqueue!(body1, agent1_position)
    tile_map[BODY1, agent1_position] = true

    agent2_position = GW.sample_empty_position(rng, tile_map)
    tile_map[AGENT2, agent2_position] = true
    body2 = DS.Queue{CartesianIndex{2}}()
    DS.enqueue!(body2, agent2_position)
    tile_map[BODY2, agent2_position] = true

    food_position = GW.sample_empty_position(rng, tile_map)
    tile_map[FOOD, food_position] = true

    reward1 = zero(R)
    reward2 = zero(R)
    food_reward = one(R) * 50 #zmiana na bardzo duza wartosc, chcialem zobaczyc czy waz w ogole lapie cos
    terminal_reward = convert(R, height * width)
    terminal_penalty = convert(R, -height * width)
    done = false

    #true, true na koncu jako avlive1, alive2
    env = Snake(tile_map, agent1_position, agent2_position, reward1, reward2, rng, done, terminal_reward, terminal_penalty, food_reward, food_position, body1, body2, true, true)

    GW.reset!(env)

    return env
end

function GW.reset!(env::Snake)
    tile_map = env.tile_map
    rng = env.rng
    body1 = env.body1
    body2 = env.body2

    tile_map[AGENT1, env.agent1_position] = false
    tile_map[AGENT2, env.agent2_position] = false
    tile_map[FOOD, env.food_position] = false

    for i in 1:length(body1)
        pos = DS.dequeue!(body1)
        tile_map[BODY1, pos] = false
    end

    for i in 1:length(body2)
        pos = DS.dequeue!(body2)
        tile_map[BODY2, pos] = false
    end

    new_agent1_position = GW.sample_empty_position(rng, tile_map)
    env.agent1_position = new_agent1_position
    tile_map[AGENT1, new_agent1_position] = true
    DS.enqueue!(body1, new_agent1_position)
    tile_map[BODY1, new_agent1_position] = true

    new_agent2_position = GW.sample_empty_position(rng, tile_map)
    env.agent2_position = new_agent2_position
    tile_map[AGENT2, new_agent2_position] = true
    DS.enqueue!(body2, new_agent2_position)
    tile_map[BODY2, new_agent2_position] = true

    new_food_position = GW.sample_empty_position(rng, tile_map)
    env.food_position = new_food_position
    tile_map[FOOD, new_food_position] = true

    env.reward1 = zero(env.reward1)
    env.reward2 = zero(env.reward2)
    env.done = false
    env.alive1 = true
    env.alive2 = true

    return nothing
end
function GW.act!(env::Snake, action, agent)
    @assert action in Base.OneTo(NUM_ACTIONS) "Invalid action $(action). Action must be in Base.OneTo($(NUM_ACTIONS))"
    @assert agent in (1, 2) "Invalid agent $(agent). Agent must be 1 or 2"

    tile_map = env.tile_map
    rng = env.rng
    body1 = env.body1
    body2 = env.body2
    _, height, width = size(tile_map)
    agent_position = agent == 1 ? env.agent1_position : env.agent2_position
    agent_body = agent == 1 ? body1 : body2

    new_agent_position = if action == 1
        GW.move_up(agent_position)
    elseif action == 2
        GW.move_down(agent_position)
    elseif action == 3
        GW.move_left(agent_position)
    else
        GW.move_right(agent_position)
    end

    snake_died = false

    #teraz sprtwadzamy czy waz umiera, gra konczy sie dopiero jak oba weze umra, zgodnie z wskazowkami z konspektu 
    if tile_map[WALL, new_agent_position] || tile_map[BODY1, new_agent_position] || tile_map[BODY2, new_agent_position]
        if agent == 1
            env.reward1 += env.terminal_penalty
            env.alive1 = false
        else
            env.reward2 += env.terminal_penalty
            env.alive2 = false
        end
        snake_died = true
    elseif tile_map[FOOD, new_agent_position]
        tile_map[agent == 1 ? AGENT1 : AGENT2, agent_position] = false
        if agent == 1
            env.agent1_position = new_agent_position
        else
            env.agent2_position = new_agent_position
        end
        tile_map[agent == 1 ? AGENT1 : AGENT2, new_agent_position] = true

        DS.enqueue!(agent_body, new_agent_position)
        tile_map[agent == 1 ? BODY1 : BODY2, new_agent_position] = true

        if length(agent_body) > 4
            last_position = DS.dequeue!(agent_body)
            tile_map[agent == 1 ? BODY1 : BODY2, last_position] = false
        end

        tile_map[FOOD, new_agent_position] = false

        if length(body1) + length(body2) == (height - 2) * (width - 2)
            if agent == 1
                env.reward1 += env.food_reward + env.terminal_reward
            else
                env.reward2 += env.food_reward + env.terminal_reward
            end
            env.done = true
        else
            new_food_position = GW.sample_empty_position(rng, tile_map)
            env.food_position = new_food_position
            tile_map[FOOD, new_food_position] = true

            if agent == 1
                env.reward1 += env.food_reward
            else
                env.reward2 += env.food_reward
            end
        end
    else
        tile_map[agent == 1 ? AGENT1 : AGENT2, agent_position] = false
        if agent == 1
            env.agent1_position = new_agent_position
        else
            env.agent2_position = new_agent_position
        end
        tile_map[agent == 1 ? AGENT1 : AGENT2, new_agent_position] = true

        DS.enqueue!(agent_body, new_agent_position)
        tile_map[agent == 1 ? BODY1 : BODY2, new_agent_position] = true

        if length(agent_body) > 4
            last_position = DS.dequeue!(agent_body)
            tile_map[agent == 1 ? BODY1 : BODY2, last_position] = false
        end

        if agent == 1
            env.reward1 -= 0.1  # kazdy ruch obarczony jest kara
            env.reward1 -= (abs(env.agent1_position[1] - env.food_position[1]) + abs(env.agent1_position[2] - env.food_position[2])) / 100.0  # oddalanie sie od jedzenia jest karane
            env.reward1 += (min(env.agent1_position[1], height - env.agent1_position[1]) + min(env.agent1_position[2], width - env.agent1_position[2])) / 100.0  # oddalanie sie od sciany jest nagradzane
        else
            env.reward2 -= 0.1  # jw
            env.reward2 -= (abs(env.agent2_position[1] - env.food_position[1]) + abs(env.agent2_position[2] - env.food_position[2])) / 100.0  # jw
            env.reward2 += (min(env.agent2_position[1], height - env.agent2_position[1]) + min(env.agent2_position[2], width - env.agent2_position[2])) / 100.0  # jw
        end
    end

    # jak oba umra to konczymy, czyli Done
    if !env.alive1 && !env.alive2
        env.done = true
    end

    return nothing
end






function move_agent(position, action)
    if action == 1
        return GW.move_up(position)
    elseif action == 2
        return GW.move_down(position)
    elseif action == 3
        return GW.move_left(position)
    else
        return GW.move_right(position)
    end
end



#####
##### miscellaneous
#####

GW.get_height(env::Snake) = size(env.tile_map, 2)
GW.get_width(env::Snake) = size(env.tile_map, 3)

GW.get_action_names(env::Snake) = (:MOVE_UP, :MOVE_DOWN, :MOVE_LEFT, :MOVE_RIGHT)
GW.get_object_names(env::Snake) = (:AGENT1, :WALL, :BODY1, :FOOD, :AGENT2, :BODY2)

function GW.get_pretty_tile_map(env::Snake, position::CartesianIndex{2})
    characters = ('☻', '█', 'x', '♦', '☻', 'x', '⋅')

    object = findfirst(@view env.tile_map[:, position])
    if isnothing(object)
        return characters[end]
    else
        return characters[object]
    end
end

function GW.get_pretty_sub_tile_map(env::Snake, window_size, position::CartesianIndex{2})
    tile_map = env.tile_map
    agent1_position = env.agent1_position
    agent2_position = env.agent2_position

    characters = ('☻', '█', 'x', '♦', '☻', 'x', '⋅')

    sub_tile_map1 = GW.get_sub_tile_map(tile_map, agent1_position, window_size)
    sub_tile_map2 = GW.get_sub_tile_map(tile_map, agent2_position, window_size)

    object1 = findfirst(@view sub_tile_map1[:, position])
    object2 = findfirst(@view sub_tile_map2[:, position])
    
    if isnothing(object1) && isnothing(object2)
        return characters[end]
    elseif isnothing(object1)
        return characters[object2]
    elseif isnothing(object2)
        return characters[object1]
    else
        return characters[object1]
    end
end


function Base.show(io::IO, ::MIME"text/plain", env::Snake)
    str = "tile_map:\n"
    str = str * GW.get_pretty_tile_map(env)
    str = str * "\nsub_tile_map:\n"
    str = str * GW.get_pretty_sub_tile_map(env, GW.get_window_size(env))
    str = str * "\nreward: $(env.reward1)"
    str = str * "\nreward: $(env.reward2)"
    str = str * "\n body1 : $(env.body1)"
    str = str * "\n body2 : $(env.body2)"
    str = str * "\ndone: $(env.done)"
    str = str * "\naction_names: $(GW.get_action_names(env))"
    str = str * "\nobject_names: $(GW.get_object_names(env))"
    len = length(env.body1)
    print(io, str, len)
    return nothing
end

GW.get_action_keys(env::Snake) = ('w', 's', 'a', 'd')

#####
##### RLBase API
#####

RLBase.StateStyle(env::GW.RLBaseEnv{E}) where {E <: Snake} = RLBase.Observation{Any}()
RLBase.state_space(env::GW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: Snake} = nothing
RLBase.state(env::GW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: Snake} = env.env.tile_map

RLBase.reset!(env::GW.RLBaseEnv{E}) where {E <: Snake} = GW.reset!(env.env)

RLBase.action_space(env::GW.RLBaseEnv{E}) where {E <: Snake} = Base.OneTo(NUM_ACTIONS)
(env::GW.RLBaseEnv{E})(action1, action2) where {E <: Snake} = GW.act!(env.env, action1, action2)

RLBase.reward(env::GW.RLBaseEnv{E}, agent) where {E <: Snake} = agent == 1 ? env.env.reward1 : env.env.reward2
RLBase.is_terminated(env::GW.RLBaseEnv{E}) where {E <: Snake} = env.env.done



function run_predefined_strategy!(env::Snake; max_steps::Int = 100)
    for _ in 1:max_steps
        # Replace with your predefined strategy logic here
        action1 = rand(1:NUM_ACTIONS)
        action2 = rand(1:2)
        GW.act!(env, action1, action2)
        

        
    end
end


end # module
