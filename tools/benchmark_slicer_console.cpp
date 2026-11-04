// Windows-only console shim for locally built Orca and installed Bambu DLLs.
// Place beside the slicer DLL and resources. It leaves the slicing engine intact
// and makes CLI diagnostics capturable when the shipped GUI shim has no console.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <filesystem>

int wmain(int argc, wchar_t** argv)
{
    wchar_t executable[32768] = {};
    if (!GetModuleFileNameW(nullptr, executable, 32768)) return 2;
    const auto directory = std::filesystem::path(executable).parent_path();
    const bool orca = std::filesystem::exists(directory / L"OrcaSlicer.dll");
    const auto library = directory / (orca ? L"OrcaSlicer.dll" : L"BambuStudio.dll");
    const auto python = directory / L"python";
    if (std::filesystem::is_directory(python)) SetDllDirectoryW(python.c_str());
    HMODULE module = LoadLibraryExW(library.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        std::fwprintf(stderr, L"Cannot load %ls: Windows error %lu\n", library.c_str(), GetLastError());
        return 3;
    }
    using MainFunction = int(__stdcall*)(int, wchar_t**);
    auto slicer_main = reinterpret_cast<MainFunction>(GetProcAddress(module, orca ? "orcaslicer_main" : "bambustu_main"));
    if (!slicer_main) {
        std::fprintf(stderr, "Slicer entry point missing: Windows error %lu\n", GetLastError());
        return 4;
    }
    return slicer_main(argc, argv);
}
