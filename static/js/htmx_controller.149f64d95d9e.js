// htmx controller.js

import { initFullscreen, updateFullscreen } from "./full_screen_controller.js";
import {
  initDarkMode,
  updateThemeToggleIcons,
} from "./dark_mode_controller.js";

document.addEventListener("DOMContentLoaded", function () {
  initFullscreen();
  initDarkMode();
});

document.body.addEventListener("htmx:afterSwap", function (event) {
  updateFullscreen();

  const currentTheme = localStorage.getItem("color-theme");
  const darkModeEnabled = currentTheme === "dark";
  updateThemeToggleIcons(darkModeEnabled);
});
