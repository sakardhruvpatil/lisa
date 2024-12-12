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
    },
    description: {
      fontSize: '18px',
      color: '#555',
      lineHeight: '1.5',
      marginBottom: '40px',
      fontStyle: 'Roboto',
      textAlign: 'left',
    },
    details: {
      textAlign: 'left',
      margin: '0 auto',
      fontSize: '18px',
      color: '#555',
      lineHeight: '2',
      maxWidth: '500px',
    },
    detailItem: {
      marginBottom: '20px',
      position: 'relative',
      paddingLeft: '20px',
      transition: 'color 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
    },
    detailItemHover: {
      color: '#fc981e',
      
    },
    footer: {
      fontSize: '18px',
      color: '#555',
      marginTop: '30px',
      textAlign: 'left',
    },
  };

  const [hovered, setHovered] = React.useState({ email: false, phone: false, address: false });

  const toggleHover = (key) => {
    setHovered((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>Contact Us</h1>
      <p style={styles.description}>
        We value your feedback and inquiries. Our team is here to assist you with any questions or concerns you may have.
      </p>
      <div style={styles.details}>
        <p
          style={{
            ...styles.detailItem,
            ...(hovered.email && styles.detailItemHover),
          }}
          onMouseEnter={() => toggleHover('email')}
          onMouseLeave={() => toggleHover('email')}
        >
          <strong>Email:</strong> contact@sakarrobotics.com
        </p>
        <p
          style={{
            ...styles.detailItem,
            ...(hovered.phone && styles.detailItemHover),
          }}
          onMouseEnter={() => toggleHover('phone')}
          onMouseLeave={() => toggleHover('phone')}
        >
          <strong>Phone:</strong> +91 79722 51272
        </p>
        <p
          style={{
            ...styles.detailItem,
            ...(hovered.address && styles.detailItemHover),
          }}
          onMouseEnter={() => toggleHover('address')}
          onMouseLeave={() => toggleHover('address')}
        >
          <strong>Address:</strong> Sakar Robotics, 2nd Floor, Pune
        </p>
      </div>
      <p style={styles.footer}>
        Whether you have a question about our services, pricing, or anything else, we are here to help you.
      </p>
    </div>
  );
};

export default Contact;
