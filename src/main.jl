include("gridworlds.jl")

using .GridWorlds

function main()
    env = GridWorlds.SnakeModule.Snake()

    println("Choose an option:")
    println("1. Play manually")
    println("2. Run predefined strategy")

    choice = readline()

    if choice == "1"
        GridWorlds.play!(env)
    elseif choice == "2"
        println("Running predefined strategy...")
        GridWorlds.predefined_play!(env)
    else
        println("Invalid choice.")
    end
end

main()
