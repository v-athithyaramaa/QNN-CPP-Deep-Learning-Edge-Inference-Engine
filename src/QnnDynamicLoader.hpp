#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <stdexcept>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace qnn_engine
{

    using QnnInterfaceGetProvidersFn_t = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
    using QnnSystemInterfaceGetProvidersFn_t = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *);

    class QnnDynamicLoader
    {
    public:
        QnnDynamicLoader() = default;

        ~QnnDynamicLoader()
        {
            unload();
        }

        QnnDynamicLoader(const QnnDynamicLoader &) = delete;
        QnnDynamicLoader &operator=(const QnnDynamicLoader &) = delete;

        QnnDynamicLoader(QnnDynamicLoader &&other) noexcept
        {
            moveFrom(std::move(other));
        }

        QnnDynamicLoader &operator=(QnnDynamicLoader &&other) noexcept
        {
            if (this != &other)
            {
                unload();
                moveFrom(std::move(other));
            }
            return *this;
        }

        void load(const std::string &backendPath, const std::string &systemPath)
        {
            // 1. Load Backend DLL (QnnCpu.dll)
            m_backendLib = LoadLibraryA(backendPath.c_str());
            if (!m_backendLib)
            {
                DWORD err = GetLastError();
                throw std::runtime_error("Failed to load QNN Backend DLL at " + backendPath +
                                         " (Win32 Error: " + std::to_string(err) + ")");
            }

            // 2. Resolve Backend Providers
            auto getBackendProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
                GetProcAddress(m_backendLib, "QnnInterface_getProviders"));
            if (!getBackendProviders)
            {
                throw std::runtime_error("Symbol 'QnnInterface_getProviders' missing in " + backendPath);
            }

            const QnnInterface_t **backendProviders = nullptr;
            uint32_t numBackendProviders = 0;
            if (getBackendProviders(&backendProviders, &numBackendProviders) != QNN_SUCCESS || numBackendProviders == 0)
            {
                throw std::runtime_error("Failed to acquire QNN Backend Interface provider");
            }
            m_backendInterface = backendProviders[0];

            // 3. Load QnnSystem DLL
            m_systemLib = LoadLibraryA(systemPath.c_str());
            if (!m_systemLib)
            {
                DWORD err = GetLastError();
                throw std::runtime_error("Failed to load QNN System DLL at " + systemPath +
                                         " (Win32 Error: " + std::to_string(err) + ")");
            }

            // 4. Resolve System Providers
            auto getSystemProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn_t>(
                GetProcAddress(m_systemLib, "QnnSystemInterface_getProviders"));
            if (!getSystemProviders)
            {
                throw std::runtime_error("Symbol 'QnnSystemInterface_getProviders' missing in " + systemPath);
            }

            const QnnSystemInterface_t **systemProviders = nullptr;
            uint32_t numSystemProviders = 0;
            if (getSystemProviders(&systemProviders, &numSystemProviders) != QNN_SUCCESS || numSystemProviders == 0)
            {
                throw std::runtime_error("Failed to acquire QNN System Interface provider");
            }
            m_systemInterface = systemProviders[0];

            std::cout << "[QnnDynamicLoader] Successfully loaded backend [" << backendPath
                      << "] and system interface [" << systemPath << "]\n";
        }

        void unload()
        {
            m_backendInterface = nullptr;
            m_systemInterface = nullptr;

            if (m_backendLib)
            {
                FreeLibrary(m_backendLib);
                m_backendLib = nullptr;
            }
            if (m_systemLib)
            {
                FreeLibrary(m_systemLib);
                m_systemLib = nullptr;
            }
        }

        [[nodiscard]] const QnnInterface_t *backend() const { return m_backendInterface; }
        [[nodiscard]] const QnnSystemInterface_t *system() const { return m_systemInterface; }

    private:
        void moveFrom(QnnDynamicLoader &&other) noexcept
        {
            m_backendLib = other.m_backendLib;
            m_systemLib = other.m_systemLib;
            m_backendInterface = other.m_backendInterface;
            m_systemInterface = other.m_systemInterface;

            other.m_backendLib = nullptr;
            other.m_systemLib = nullptr;
            other.m_backendInterface = nullptr;
            other.m_systemInterface = nullptr;
        }

        HMODULE m_backendLib{nullptr};
        HMODULE m_systemLib{nullptr};
        const QnnInterface_t *m_backendInterface{nullptr};
        const QnnSystemInterface_t *m_systemInterface{nullptr};
    };

}