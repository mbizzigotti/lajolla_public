
basic.slang.inl: basic.spv
	xxd -i basic.spv > basic.slang.inl

basic.spv: basic.slang
	slangc basic.slang -target spirv -o basic.spv
