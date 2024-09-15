include("gridworlds.jl")
include("TDLearning.jl")

using .GridWorlds
using .TDLearning

# Inicjalizacja środowiska gry
env = GridWorlds.SnakeModule.Snake()

# Inicjalizacja agenta TD-learning
learner = TDLearning.TDLearner()

# Uruchomienie nauki TD-learning, zapis do csv i liczna ezpizodow
TDLearning.run_td_learning!(env, learner, 2000, "test.csv")  # Liczba epizodów do nauki
