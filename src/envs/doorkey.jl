export DoorKey

using Random

mutable struct DoorKey{W<:GridWorldBase, R} <: AbstractGridWorld
    world::W
    agent::Agent
    goal_reward::Float64
    reward::Float64
    rng::R
end

function DoorKey(; n = 7, agent_start_pos = CartesianIndex(2,2), agent_start_dir = RIGHT, goal_pos = CartesianIndex(n-1, n-1), rng = Random.GLOBAL_RNG)
    door = Door(:yellow)
    key = Key(:yellow)

    objects = (EMPTY, WALL, GOAL, door, key)

    world = GridWorldBase(objects, n, n)

    world[WALL, [1,n], 1:n] .= true
    world[WALL, 1:n, [1,n]] .= true
    world[GOAL, goal_pos] = true

    goal_reward = 1.0
    reward = 0.0

    env = DoorKey(world, Agent(dir = agent_start_dir, pos = agent_start_pos), goal_reward, reward, rng)

    reset!(env, agent_start_pos = agent_start_pos, agent_start_dir = agent_start_dir, goal_pos = goal_pos)

    return env
end

function (env::DoorKey)(::MoveForward)
    world = get_world(env)
    objects = get_objects(env)
    agent = get_agent(env)

    set_reward!(env, 0.0)

    door = objects[end - 1]
    key = objects[end]

    dir = get_agent_dir(env)
    dest = dir(get_agent_pos(env))

    if world[key, dest]
        if PICK_UP(agent, key)
            world[key, dest] = false
            world[EMPTY, dest] = true
        end
        set_agent_pos!(env, dest)
    elseif world[door, dest] && agent.inventory !== key
        nothing
    elseif world[door, dest] && agent.inventory === key
        set_agent_pos!(env, dest)
    elseif !world[WALL,dest]
        set_agent_pos!(env, dest)
        if world[GOAL, get_agent_pos(env)]
            set_reward!(env, env.goal_reward)
        end
    end

    return env
end

RLBase.get_terminal(env::DoorKey) = get_world(env)[GOAL, get_agent_pos(env)]

function RLBase.reset!(env::DoorKey; agent_start_pos = CartesianIndex(2, 2), agent_start_dir = RIGHT, goal_pos = CartesianIndex(get_width(env) - 1, get_width(env) - 1))
    world = get_world(env)
    n = get_width(env)

    objects = get_objects(env)
    door = objects[end - 1]
    key = objects[end]

    set_reward!(env, 0.0)

    set_agent_pos!(env, agent_start_pos)

    set_agent_dir!(env, agent_start_dir)

    door_pos = CartesianIndex(rand(env.rng, 2:n-1), rand(env.rng, 3:n-2))
    @assert agent_start_pos[2] < door_pos[2] "Agent should start on the left side of the door"
    @assert goal_pos[2] > door_pos[2] "Goal should be placed on the right side of the door"
    world[WALL, :, door_pos[2]] .= true
    world[door, door_pos] = true
    world[WALL, door_pos] = false

    key_pos = CartesianIndex(rand(env.rng, 2:n-1), rand(env.rng, 2:door_pos[2]-1))
    while key_pos == agent_start_pos
        key_pos = CartesianIndex(rand(env.rng, 2:n-1), rand(env.rng, 2:door_pos[2]-1))
    end
    world[key, key_pos] = true

    world[EMPTY, :, :] .= .!(.|((world[x, :, :] for x in [WALL, GOAL, door, key])...))
end
