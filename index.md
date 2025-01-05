---
layout: default
title: "Ziyad BENKHADAJ"
---

en cours de construction...
loading...

<!-- Language Selector (Flags for English and French) -->
<div id="language-toggle">
  <button onclick="setLanguage('en')">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg" alt="English" style="width: 30px; height: 20px;">
    English
  </button>
  <button onclick="setLanguage('fr')">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Flag_of_France.svg" alt="Français" style="width: 30px; height: 20px;">
    Français
  </button>
</div>

<!-- Image of Ziyad (Uploaded locally) -->
<div class="profile-image">
  <img src="{{ site.baseurl }}/assets/photo_Ziyad_Alpes.JPG" alt="Ziyad BENKHADAJ" style="width: 200px; height: auto;">
</div>

<!-- English Content (Default hidden) -->
<div id="content-en" class="language-content">
  <h1>Welcome to my website</h1>
  <p>This is my personal website where you can find various information about me and the services I offer.</p>
  
  <ul>
    <li>CV</li>
    <li>Philosophical opinions</li>
    <li>Math and physics tutoring, from high school to higher education (preparatory classes, university, medical school...)</li>
  </ul>

  <h2>For contact, please email me at:</h2>
  <p>Email: <a href="mailto:ziyad.benkhadaj@gmail.com">ziyad.benkhadaj@gmail.com</a></p>
  
</div>

<!-- French Content (Default hidden) -->
<div id="content-fr" class="language-content" style="display:none;">
  <h1>Bienvenue sur mon site web</h1>
  <p>Ceci est mon site personnel où vous pouvez trouver diverses informations sur moi et les services que je propose.</p>
  
  <ul>
    <li>CV</li>
    <li>Opinions philosophiques</li>
    <li>Soutien scolaire en mathématiques et physique, du lycée jusqu'au supérieur (prépa, licence, médecine...)</li>
  </ul>

  <h2>Pour me contacter, merci de m'écrire à l'adresse suivante:</h2>
  <p>Email : <a href="mailto:ziyad.benkhadaj@gmail.com">ziyad.benkhadaj@gmail.com</a></p>
  
</div>



<!-- Real-Time Date and Time -->
<div id="datetime" style="font-size: 18px; font-family: Arial, sans-serif; color: white;">
  
</div>

<!-- JavaScript to Toggle Language and Display Date/Time -->
<script>
  var currentLanguage = 'en'; // Default language is English

  function setLanguage(lang) {
    currentLanguage = lang; // Update the global currentLanguage variable

    // Hide both language contents
    document.getElementById('content-en').style.display = 'none';
    document.getElementById('content-fr').style.display = 'none';

    // Show the selected language content
    if (lang === 'en') {
      document.getElementById('content-en').style.display = 'block';
    } else if (lang === 'fr') {
      document.getElementById('content-fr').style.display = 'block';
    }

    // Update the date/time based on language
    updateDateTime(lang);
  }

  function updateDateTime(lang) {
    var currentDate = new Date();
    var options = { 
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', 
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true 
    };
    var formattedDate = currentDate.toLocaleDateString('en-US', options); // Default to English format

    if (lang === 'fr') {
      formattedDate = currentDate.toLocaleDateString('fr-FR', options); // French date format
    }

    // Update the date/time in the corresponding section
    if (lang === 'en') {
      document.getElementById('current-date-time-en').textContent = formattedDate;
    } else if (lang === 'fr') {
      document.getElementById('current-date-time-fr').textContent = formattedDate;
    }
  }

  // Function to continuously update the real-time date and time
  function updateRealTime() {
    var currentDate = new Date();
    var options = { 
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', 
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true 
    };

    // If language is French, change the hour format to 24-hour
  if (currentLanguage === 'fr') {
    options.hour12 = false;  // Switch to 24-hour format for French
  }
    
    // Display the real-time date in the selected language
    var formattedDate = currentDate.toLocaleDateString(currentLanguage === 'fr' ? 'fr-FR' : 'en-US', options);

    // Update the real-time date in the selected language sections
    document.getElementById('datetime').textContent = formattedDate;
    document.getElementById('current-date-time-en').textContent = formattedDate;
    document.getElementById('current-date-time-fr').textContent = formattedDate;
  }

  // Set the real-time update interval to 1 second (1000 milliseconds)
  setInterval(updateRealTime, 1000);

  // Set default language to English and update the date/time
  setLanguage('fr');
</script>
