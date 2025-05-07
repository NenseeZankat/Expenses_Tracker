import { Fragment, useState, useEffect } from "react";
import { Disclosure } from "@headlessui/react";
import { Bars3Icon, BellIcon, XMarkIcon, SunIcon, MoonIcon } from "@heroicons/react/24/outline";
import { Link, useLocation } from "react-router-dom";
import { useDispatch } from "react-redux";
import { IoLogOutOutline } from "react-icons/io5";
import { logoutAction } from "../../redux/slice/authSlice";

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

export default function PrivateNavbar() {
  const dispatch = useDispatch();
  const location = useLocation();
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("theme") === "dark";
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (darkMode) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  const logoutHandler = () => {
    dispatch(logoutAction());
    localStorage.removeItem("userInfo");
  };

  const navItems = [
    { name: "Dashboard", to: "/dashboard" },
    { name: "Add Transaction", to: "/add-transaction" },
    { name: "Add Purchase", to: "/add-purchase" },
    { name: "Add Category", to: "/add-category" },
    { name: "Categories", to: "/categories" },
    { name: "Profile", to: "/profile" },
    { name: "Family", to: "/family" },
    { name: "Create Family", to: "/create-family" },
    { name: "Create Family Goal", to: "/familyGoalForm" },
    { name: "Manage Subscriptions", to: "/manage-subscriptions" },
    { name: "Add Budget", to: "/budget" },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <Disclosure as="nav" className="bg-white dark:bg-gray-900 shadow">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 justify-between items-center">
              {/* Left: Logo and Nav Links */}
              <div className="flex items-center w-full">
                <div className="flex-shrink-0 text-green-500">
                  <img className="h-10 w-auto" src="/Expenza.png" alt="Expenza Logo" />
                </div>
                {/* Desktop Menu */}
                <div className="hidden md:flex md:space-x-4 ml-6 w-full items-center">
                  {navItems.map((item) => (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={classNames(
                        isActive(item.to)
                          ? "border-indigo-500 text-indigo-600 dark:text-white"
                          : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 dark:hover:text-white",
                        "inline-flex items-center border-b-2 px-1 pt-1 text-sm font-medium"
                      )}
                    >
                      {item.name}
                    </Link>
                  ))}
                </div>
              </div>

              {/* Right: Hamburger + Logout */}
              <div className="flex items-center gap-2">
                {/* Mobile Hamburger Menu */}
                <div className="flex items-center md:hidden">
                  <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-500 dark:hover:bg-gray-700">
                    {open ? (
                      <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                    ) : (
                      <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                    )}
                  </Disclosure.Button>
                </div>

                {/* Logout Button */}
                <button
                  onClick={logoutHandler}
                  className="inline-flex items-center gap-1.5 rounded-md bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-700"
                >
                  <IoLogOutOutline className="h-5 w-5" />
                  Logout
                </button>
              </div>
            </div>
          </div>

          {/* Mobile Menu */}
          <Disclosure.Panel className="md:hidden bg-white dark:bg-gray-800 max-h-[80vh] overflow-y-auto">
            <div className="space-y-1 px-2 pt-2 pb-3">
              {navItems.map((item) => (
                <Link key={item.to} to={item.to}>
                  <Disclosure.Button
                    as="span"
                    className={classNames(
                      isActive(item.to)
                        ? "bg-indigo-50 border-indigo-500 text-indigo-700 dark:bg-gray-700 dark:text-white"
                        : "border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800 dark:text-gray-300 dark:hover:text-white",
                      "block border-l-4 py-2 pl-3 pr-4 text-base font-medium"
                    )}
                  >
                    {item.name}
                  </Disclosure.Button>
                </Link>
              ))}
              <Disclosure.Button
                as="button"
                onClick={logoutHandler}
                className="w-full text-left border-l-4 border-transparent py-2 pl-3 pr-4 text-base font-medium text-red-600 hover:bg-red-100 dark:hover:bg-red-600 dark:text-white"
              >
                Sign out
              </Disclosure.Button>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
