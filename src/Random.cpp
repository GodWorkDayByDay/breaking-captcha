/**
 * \file Random.cpp
 * \author Guy Rutenberg <guyrutenberg@gmail.com>
 * \version 1.0
 * \date 2007
 *
 * Header file for Random.
 *
 * License: The MIT License (included in the file).
*/
 
/*****************************************************************************
The MIT License
 
Copyright (c) 2007 Guy Rutenberg
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*****************************************************************************/

#include "Random.h"

Random::Random()
{
	m_random.open("/dev/random",std::ios::in|std::ios::binary);
	m_urandom.open("/dev/urandom",std::ios::in|std::ios::binary);
	if (!m_random || !m_urandom)
		throw DEVRANDOMERROR;
}

unsigned int Random::secure()
{
	unsigned int num = 0;
	m_random.read((char*)&num, sizeof(unsigned int));
	return num;
}

unsigned int Random::strong()
{
	unsigned int num = 0;
	m_urandom.read((char*)&num, sizeof(unsigned int));
	return num;
}

double Random::secure_real()
{
	return (double)this->secure()/(unsigned int)(-1);
}

double Random::strong_real()
{
	return (double)this->strong()/(unsigned int)(-1);
}

unsigned int Random::secure_range(unsigned int x)
{
	// we do mod x to handle the extreme case secure_real() returns 1
	return (unsigned int)(this->secure_real() * x)%x;
}

unsigned int Random::strong_range(unsigned int x)
{
	// we do mod x to handle the extreme case secure_real() returns 1
	return (unsigned int)(this->strong_real() * x)%x;
}
