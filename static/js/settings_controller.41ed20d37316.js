// settings_controller.js

export function initSettings() {
    const saveButton = document.getElementById("save-button");
  
    if (saveButton) {
      saveButton.addEventListener("click", saveChanges);
    }
  }
  
  export function saveChanges() {
    // Your save changes logic here
    // For now, we'll simply log a message to the console
    console.log("Changes saved!");
  }
  