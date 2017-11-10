#!/usr/bin/env julia

include("src/MLToolkit.jl")

# only run anything if it's being run from the command line
if !isinteractive()
	MLToolkit.run()
end
