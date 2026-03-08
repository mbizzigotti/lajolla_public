
build: rgen_path_tracing.spv miss.spv chit_tri.spv ahit.spv

%.spv: Main.slang
	slangc Main.slang -target spirv -entry $* -o $@
