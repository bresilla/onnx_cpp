set_languages("cxx17")
add_rules("mode.debug", "mode.release")

add_requires("cmake::OpenCV", {system = true, alias = "opencv", configs = {moduledirs = "/usr/local/share/cmake"}})
add_requires("onnxruntime", {system = true, configs = {moduledirs = "/usr/local/share/cmake"}})
-- add_requires("tensorrt", {system = true})
add_requires("spdlog")


target("detector")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/main.cpp")
    add_files("src/detections.cpp")
    add_files("src/onnxinf.cpp")
    add_files("src/opencvinf.cpp")
    -- add_files("src/tensorinf.cpp")
    add_packages("opencv")
    add_packages("spdlog")
    add_packages("onnxruntime")
