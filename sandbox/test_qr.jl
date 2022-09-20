### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ 464a59d0-1827-4500-8f7f-9f57980b64c3
using SparseArrays, LinearAlgebra

# ╔═╡ 38db5050-0f16-4b98-badf-49f402ba901e
using JLD

# ╔═╡ 8eb6a537-8c7a-4d9d-be22-d97d679bf2d7
using Profile

# ╔═╡ 3505520a-995a-4827-b07a-7958beac0973
A = let
	A = JLD.load("/Users/degleris/Data/big_A.jld", "A")
	A = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval)
	A = sparse(A')
end

# ╔═╡ 0afe7126-3343-43c8-8128-7c9f9131d0c4
@elapsed F = qr(A)

# ╔═╡ 05a588c7-6fc8-4b02-824f-043246413ef3
Ap = A[F.prow, F.pcol]

# ╔═╡ 95579532-d3a0-4cb4-b7e8-c8239ab35f18
SparseArrays.sparse!

# ╔═╡ 13b01d36-eb60-464e-8f46-3b40b0ed2f77
function mgs(A)
	m, n = size(A)

	V = [A[:, j] for j in 1:n]
	R = spzeros(n, n)

	cnt = 0

	for i = 1:n
		R[i, i] = norm(V[i])
		V[i] /= R[i, i]

		for j = i+1:n  # CAN BE PARALLELIZED
			rij = dot(V[i], V[j])
			R[i, j] = rij
			
			if rij != 0
				cnt += 1
				
				#Q[:, j] -= R[i, j] * view(Q, :, i)
				LinearAlgebra.axpy!(-rij, V[i], V[j])
			end
		end
	end
				
	return V, R, cnt
end

# ╔═╡ 4daefd22-b64b-47ff-ab38-571fd720cb06
A_small = Ap[:, 1:5000];

# ╔═╡ 7c2cecba-ca7f-4664-96f1-b724b081410f
V, R, cnt = mgs(A_small);

# ╔═╡ 97ec3c9d-977b-46bb-b799-533e5b7044d5
Q = hcat(V...);

# ╔═╡ b0002874-5dc6-48b2-a824-ce3fe5c8dded
cnt

# ╔═╡ 526ac76f-f938-4486-8fde-512249e3e537
norm(Q'Q - I)

# ╔═╡ 78d0caa6-f232-4d9d-bf5f-c1226485ef29
nnz(R) / nnz(A_small)

# ╔═╡ 30e10262-21a9-4a0c-9c66-6d4d2c9258f3
nnz(Q) / nnz(A_small)

# ╔═╡ 36a5d8d1-ada2-4230-a21f-0a0e337b3cb2
# begin
# 	Profile.clear()
# 	@profile mgs(A_small)
# end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JLD = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
JLD = "~0.13.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "310b77648d38c223d947ff3f50f511d08690b8d5"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.3"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "91d6baa911283650df649d0aea7c28639273ae7b"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.1+0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "94f5101b96d2d968ace56f7f2db19d0a5f592e28"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.15.0"

[[deps.H5Zblosc]]
deps = ["Blosc", "HDF5"]
git-tree-sha1 = "26b22c9039b039e29ec4f4f989946de722e87ab9"
uuid = "c8ec2601-a99c-407f-b158-e79c03c2f5f7"
version = "0.1.0"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "899f041bf330ebeead3637073b2ca7477760edde"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.11"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JLD]]
deps = ["Compat", "FileIO", "H5Zblosc", "HDF5", "Printf"]
git-tree-sha1 = "cd46c18390e9bbc37a2098dfb355ec5f18931900"
uuid = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
version = "0.13.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═464a59d0-1827-4500-8f7f-9f57980b64c3
# ╠═38db5050-0f16-4b98-badf-49f402ba901e
# ╠═8eb6a537-8c7a-4d9d-be22-d97d679bf2d7
# ╠═3505520a-995a-4827-b07a-7958beac0973
# ╠═0afe7126-3343-43c8-8128-7c9f9131d0c4
# ╠═05a588c7-6fc8-4b02-824f-043246413ef3
# ╠═95579532-d3a0-4cb4-b7e8-c8239ab35f18
# ╠═13b01d36-eb60-464e-8f46-3b40b0ed2f77
# ╠═4daefd22-b64b-47ff-ab38-571fd720cb06
# ╠═7c2cecba-ca7f-4664-96f1-b724b081410f
# ╠═97ec3c9d-977b-46bb-b799-533e5b7044d5
# ╠═b0002874-5dc6-48b2-a824-ce3fe5c8dded
# ╠═526ac76f-f938-4486-8fde-512249e3e537
# ╠═78d0caa6-f232-4d9d-bf5f-c1226485ef29
# ╠═30e10262-21a9-4a0c-9c66-6d4d2c9258f3
# ╠═36a5d8d1-ada2-4230-a21f-0a0e337b3cb2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
