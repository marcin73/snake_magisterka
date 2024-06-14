const ESC = Char(0x1B)
const HIDE_CURSOR = ESC * "[?25l"
const SHOW_CURSOR = ESC * "[?25h"
const CLEAR_SCREEN = ESC * "[2J"
const MOVE_CURSOR_TO_ORIGIN = ESC * "[H"
const CLEAR_SCREEN_BEFORE_CURSOR = ESC * "[1J"
const EMPTY_SCREEN = CLEAR_SCREEN_BEFORE_CURSOR * MOVE_CURSOR_TO_ORIGIN
const DEFAULT_FRAME_START_DELIMITER = "FRAME_START_DELIMITER"

function open_maybe(file_name::AbstractString)
    if isfile(file_name)
        error("File $(file_name) already exists!")
    else
        open(file_name, "w")
    end
end

open_maybe(::Nothing) = nothing

close_maybe(io::IO) = close(io)
close_maybe(io::Nothing) = nothing

write_maybe(io::IO, content) = write(io, content)
write_maybe(io::Nothing, content) = 0

function play!(terminal::REPL.Terminals.UnixTerminal, env::AbstractGridWorld, file_name::Union{Nothing, AbstractString}, frame_start_delimiter::AbstractString)
    terminal_out = terminal.out_stream
    terminal_in = terminal.in_stream
    file = open_maybe(file_name)

    action_keys_agent1 = ('w', 's', 'a', 'd')
    action_keys_agent2 = ('i', 'k', 'j', 'l')
    key_bindings = "Key bindings to play: 'q': quit, 'r': GW.reset!, $(action_keys_agent1): agent1 act, $(action_keys_agent2): agent2 act"

    char = nothing

    write(terminal_out, CLEAR_SCREEN)
    write(terminal_out, MOVE_CURSOR_TO_ORIGIN)
    write(terminal_out, HIDE_CURSOR)

    REPL.Terminals.raw!(terminal, true)

    try
        while true
            write_maybe(file, frame_start_delimiter)

            frame = key_bindings
            frame = frame * "\n" * "Last play character read: $(char)"
            frame = frame * "\n" * repr(MIME"text/plain"(), env)

            write_maybe(terminal_out, frame)
            write_maybe(file, frame)

            char = read(terminal_in, Char)

            if char == 'q'
                write(terminal_out, SHOW_CURSOR)
                close_maybe(file)
                REPL.Terminals.raw!(terminal, false)
                return nothing
            elseif char == 'r'
                reset!(env)
            elseif char in action_keys_agent1
                act!(env, findfirst(==(char), action_keys_agent1), 1)
            elseif char in action_keys_agent2
                act!(env, findfirst(==(char), action_keys_agent2), 2)
            end

            write(terminal_out, EMPTY_SCREEN)
        end
    finally
        write(terminal_out, SHOW_CURSOR)
        close_maybe(file)
        REPL.Terminals.raw!(terminal, false)
    end

    return nothing
end


play!(env::AbstractGridWorld; file_name = nothing, frame_start_delimiter = DEFAULT_FRAME_START_DELIMITER) = play!(REPL.TerminalMenus.terminal, env, file_name, frame_start_delimiter)

function replay(terminal::REPL.Terminals.UnixTerminal, file_name::AbstractString, frame_start_delimiter::AbstractString, frame_rate::Union{Nothing, Real})
    terminal_out = terminal.out_stream
    strings = split(read(file_name, String), frame_start_delimiter)
    frames = @view strings[2:end]
    num_frames = length(frames)

    write(terminal_out, CLEAR_SCREEN)
    write(terminal_out, MOVE_CURSOR_TO_ORIGIN)
    write(terminal_out, HIDE_CURSOR)

    if isnothing(frame_rate)
        terminal_in = terminal.in_stream
        replay_key_bindings = "Key bindings to replay: 'q': quit, 'f': first frame, 'n': next frame, 'p': previous frame"
        current_frame = 1
        char = nothing

        REPL.Terminals.raw!(terminal, true)

        try
            while true
                replay_frame = replay_key_bindings
                replay_frame = replay_frame * "\n" * "Last replay character read: $(char)"
                replay_frame = replay_frame * "\n" * "frame number: $(current_frame)/$(num_frames)"
                replay_frame = replay_frame * "\n" * "------------FRAME_START------------"
                replay_frame = replay_frame * "\n" * frames[current_frame]

                write(terminal_out, replay_frame)

                char = read(terminal_in, Char)

                if char == 'q'
                    write(terminal_out, SHOW_CURSOR)
                    REPL.Terminals.raw!(terminal, false)
                    return nothing
                elseif char == 'f'
                    current_frame = 1
                elseif char == 'n'
                    current_frame = mod1(current_frame + 1, num_frames)
                elseif char == 'p'
                    current_frame = mod1(current_frame - 1, num_frames)
                end

                write(terminal_out, EMPTY_SCREEN)
            end
        finally
            write(terminal_out, SHOW_CURSOR)
            REPL.Terminals.raw!(terminal, false)
            return nothing
        end
    else
        write(terminal_out, CLEAR_SCREEN)
        write(terminal_out, MOVE_CURSOR_TO_ORIGIN)
        write(terminal_out, HIDE_CURSOR)

        for frame in frames
            write(terminal_out, frame)
            sleep(1 / frame_rate)
            write(terminal_out, EMPTY_SCREEN)
        end

        write(terminal_out, SHOW_CURSOR)
        return nothing
    end
end

replay(; file_name, frame_start_delimiter = DEFAULT_FRAME_START_DELIMITER, frame_rate = nothing) = replay(REPL.TerminalMenus.terminal, file_name, frame_start_delimiter, frame_rate)

function predefined_play!(terminal::REPL.Terminals.UnixTerminal, env::AbstractGridWorld, file_name::Union{Nothing, AbstractString}, frame_start_delimiter::AbstractString)
    terminal_out = terminal.out_stream
    terminal_in = terminal.in_stream
    file = open_maybe(file_name)

    action_keys_agent1 = ('w', 's', 'a', 'd')
    action_keys_agent2 = ('i', 'k', 'j', 'l')
    key_bindings = "Key bindings to play: 'q': quit, 'r': GW.reset!, $(action_keys_agent1): agent1 act, $(action_keys_agent2): agent2 act"

    char = nothing

    write(terminal_out, CLEAR_SCREEN)
    write(terminal_out, MOVE_CURSOR_TO_ORIGIN)
    write(terminal_out, HIDE_CURSOR)

    REPL.Terminals.raw!(terminal, true)

    try
        while true
            write_maybe(file, frame_start_delimiter)

            frame = key_bindings
            frame = frame * "\n" * "Last play character read: $(char)"
            frame = frame * "\n" * repr(MIME"text/plain"(), env)

            write_maybe(terminal_out, frame)
            write_maybe(file, frame)

            # Randomly select actions for each agent
            action1 = rand(1:4)
            action2 = rand(1:4)

            act!(env, action1, 1)
            act!(env, action2, 2)

            char = read(terminal_in, Char)

            if char == 'q'
                write(terminal_out, SHOW_CURSOR)
                close_maybe(file)
                REPL.Terminals.raw!(terminal, false)
                return nothing
            elseif char == 'r'
                reset!(env)
            elseif char in action_keys_agent1
                act!(env, findfirst(==(char), action_keys_agent1), 1)
            elseif char in action_keys_agent2
                act!(env, findfirst(==(char), action_keys_agent2), 2)
            end

            write(terminal_out, EMPTY_SCREEN)
        end
    finally
        write(terminal_out, SHOW_CURSOR)
        close_maybe(file)
        REPL.Terminals.raw!(terminal, false)
    end

    return nothing
end

predefined_play!(env::AbstractGridWorld; file_name = nothing, frame_start_delimiter = DEFAULT_FRAME_START_DELIMITER) = predefined_play!(REPL.TerminalMenus.terminal, env, file_name, frame_start_delimiter)
