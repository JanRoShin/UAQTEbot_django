// dark_mode_controller.js

let darkModeInitialized = false;

export function initDarkMode() {
  if (!darkModeInitialized) {
    darkModeInitialized = true;

    // Check for saved theme preference or use the system preference
    const darkModeEnabled =
      localStorage.getItem("color-theme") === "dark" ||
      (!("color-theme" in localStorage) &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);

    // Update the icon visibility and class based on the theme
    updateThemeToggleIcons(darkModeEnabled);

    // Attach the click event listener to the body (event delegation)
    document.body.addEventListener("click", handleDarkModeToggle);
  }
}

function handleDarkModeToggle(event) {
  const themeToggleButton = document.getElementById("theme-toggle");
  if (themeToggleButton && event.target === themeToggleButton) {
    // Toggle theme only if the click is on the theme toggle button
    const darkModeEnabled = document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", !darkModeEnabled);

    // Update the icon visibility and class based on the new theme
    updateThemeToggleIcons(!darkModeEnabled);
    updateLocalStorageTheme(!darkModeEnabled);
  }
}

function updateLocalStorageTheme(darkModeEnabled) {
  // Update the local storage based on the new theme
  localStorage.setItem("color-theme", darkModeEnabled ? "dark" : "light");
}

// dark_mode_controller.js
export function updateThemeToggleIcons(darkModeEnabled) {
  setTimeout(() => {
    const darkIcon = document.getElementById("theme-toggle-dark-icon");
    const lightIcon = document.getElementById("theme-toggle-light-icon");

    if (darkModeEnabled) {
      lightIcon.style.display = "inline-block";
      darkIcon.style.display = "none";
    } else {
      darkIcon.style.display = "inline-block";
      lightIcon.style.display = "none";
    }
    // Increased delay to 30 milliseconds to allow the icon to be updated
  }, 30);
}
