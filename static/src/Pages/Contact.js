import React from 'react';

const Contact = () => {
  const styles = {
    container: {
      maxWidth: '800px',
      margin: '50px auto',
      padding: '30px',
      borderRadius: '12px',
      backgroundColor: '#fff',
      boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
      fontFamily: 'Roboto, sans-serif',
      textAlign: 'center',
      overflow: 'hidden',
      transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
    },
   
    heading: {
      fontSize: '36px',
      color: '#fc981e',
      marginBottom: '20px',
      textTransform: 'uppercase',
      letterSpacing: '2px',
      transition: 'color 0.3s ease',
    },
 
    description: {
      fontSize: '18px',
      color: '#555',
      lineHeight: '1.8',
      marginBottom: '20px',
      fontStyle: 'italic',
      transition: 'all 0.3s ease-in-out',
    },
    descriptionHover: {
      color: '#333',
    },
    list: {
      listStyleType: 'none',
      padding: 0,
      margin: '20px 0',
      fontSize: '18px',
      color: '#555',
      textAlign: 'left',
    },
    listItem: {
      marginBottom: '15px',
      position: 'relative',
      paddingLeft: '25px',
      transition: 'all 0.3s ease-in-out',
    },
    listItemHover: {
      color: '#fc981e',
      transform: 'translateX(10px)',
    },
    listItemIcon: {
      content: "'📍'", // Icon as content
      position: 'absolute',
      left: 0,
      top: 0,
      fontSize: '20px',
      color: '#fc981e',
    },
  };

  // Hover effects with state
  const [hovered, setHovered] = React.useState({ container: false, heading: false, description: false, listItems: {} });

  const toggleHover = (key, index = null) => {
    setHovered((prev) => {
      if (index !== null) {
        return {
          ...prev,
          listItems: { ...prev.listItems, [index]: !prev.listItems[index] },
        };
      }
      return { ...prev, [key]: !prev[key] };
    });
  };

  return (
    <div
      style={{
        ...styles.container,
        ...(hovered.container && styles.containerHover),
      }}
      onMouseEnter={() => toggleHover('container')}
      onMouseLeave={() => toggleHover('container')}
    >
      <h1
        style={{
          ...styles.heading,
          ...(hovered.heading && styles.headingHover),
        }}
        onMouseEnter={() => toggleHover('heading')}
        onMouseLeave={() => toggleHover('heading')}
      >
        Contact Us
      </h1>
      <p
        style={{
          ...styles.description,
          ...(hovered.description && styles.descriptionHover),
        }}
        onMouseEnter={() => toggleHover('description')}
        onMouseLeave={() => toggleHover('description')}
      >
        We value your feedback and inquiries. Our team is here to assist you with any questions or concerns you may have.
      </p>
      <ul style={styles.list}>
        {['Email: contact@sakarrobotics.com', 'Phone: +91 79722 51272', 'Address: Sakar Robotics, 2nd Floor, Pune'].map((item, index) => (
          <li
            key={index}
            style={{
              ...styles.listItem,
              ...(hovered.listItems[index] && styles.listItemHover),
            }}
            onMouseEnter={() => toggleHover(null, index)}
            onMouseLeave={() => toggleHover(null, index)}
          >
            {item}
          </li>
        ))}
      </ul>
      <p
        style={{
          ...styles.description,
          ...(hovered.description && styles.descriptionHover),
        }}
      >
        Whether you have a question about our services, pricing, or anything else, we are here to help you.
      </p>
    </div>
  );
};

export default Contact;
